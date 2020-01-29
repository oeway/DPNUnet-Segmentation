import os
import random

import numpy as np
from utils import load_path_mapping, bytescale, read_model, cut_border, wsh, generate_geojson_masks, get_preprocessing, \
    download_with_url, fill_holes
from callbacks import ModelSaver, CheckpointSaver, ModelFreezer, TensorBoard, Callback, PredictionSaver
from dataset import BorderImageType, SigmoidBorderImageType, PaddedImageType, PaddedSigmoidImageType, CachingImageProvider, InFolderImageProvider, TrainDataset, ValDataset, SequentialDataset, set_output_dir
from trainer import Estimator, PytorchTrain, predict8tta
from augmentations.transforms import aug_victor, aug_binarize
from pytorch_zoo import unet
import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from configs import get_default_config
import json
import tqdm

import urllib.request
import zipfile
from distutils.dir_util import copy_tree
from imageio import imread, imwrite
from scipy.stats import pearsonr

from skimage.morphology import remove_small_objects, watershed, remove_small_holes, skeletonize, binary_closing,\
    binary_dilation, binary_erosion
from skimage import measure, exposure, segmentation
from matplotlib import cm
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy import ndimage as ndi
import cv2

import random
from matplotlib import cm
jet_colors = [cm.jet(i)[:3] for i in range(256)]
random.shuffle(jet_colors)


def wsh_nuclei_seeded(config, nuclei_output_name, cell_output_name):
    """This is for hpa_image segmentation processing
    """
    masks_files = [os.path.splitext(nuclei_output_name[0])[0] + '_output.png', os.path.splitext(cell_output_name[0])[0] + '_output.png']
    target_folder = config.outputs_dir 
    sample_folders = list(
        map(lambda item: os.path.join(target_folder, item) if not item.startswith('.') else None, os.listdir(target_folder)))
    sample_folders = list(filter(lambda item: item if item else None, sample_folders))

    jet_colors = [cm.jet(i)[:3] for i in range(256)]
    random.shuffle(jet_colors)

    for sf in sample_folders:
        print('processing ' + sf)
        cell_img = imread(os.path.join(sf, masks_files[1]))
        nuclei_img = imread(os.path.join(sf, masks_files[0]))
        nuclei_label = wsh(nuclei_img[...,2] / 255., \
            0.4, 1 - (nuclei_img[...,1]+cell_img[..., 1]) / 255. > 0.05,nuclei_img[...,2] / 255, threshold_adjustment=-0.25, \
                small_object_size_cutoff=500)

        # for hpa_image, to remove the small pseduo nuclei
        # comment, found two separate nuclei regions (neighbour) with the same value. could be imporvoved.
        nuclei_label = remove_small_objects(nuclei_label, 1500).astype(np.uint8)
        # till here
        nuclei_label_overlay = label2rgb(nuclei_label, bg_label=0, bg_color=(0.8, 0.8, 0.8), colors=jet_colors)
        # this one is carefully set to highlight the cell border signal, iteration number. and then skeletonize to avoid trailoring the cell signals
        sk = skeletonize(ndi.morphology.binary_dilation(cell_img[..., 1]/255.0>0.05, iterations=5))
        # this is to remove the cell borders' signal from cell mask. could use np.logical_and with some revision, to replace this func. Tuned for segmentation hpa images
        sk = np.subtract(np.asarray(cell_img[...,2]/255>0.2, dtype=np.int8), np.asarray(sk, dtype=np.int8))

        sk = np.clip(sk, 0, 1.0)
        distance = ndi.distance_transform_edt(sk)

        cell_label = watershed(-distance, nuclei_label, mask=sk)
        cell_label = fill_holes(cell_label)
        image_label_overlay = label2rgb(cell_label, bg_label=0, bg_color=(0.8, 0.8, 0.8), colors=jet_colors)

        cv2.imwrite(os.path.join(sf, os.path.splitext(nuclei_output_name[0])[0] + '_label.png'), nuclei_label.astype('uint16'))
        cv2.imwrite(os.path.join(sf, os.path.splitext(nuclei_output_name[0])[0] + '_output_color.png'), bytescale(nuclei_label_overlay).astype('uint8'))
        cv2.imwrite(os.path.join(sf, cell_output_name[0].split('_')[0] + '_combined_labels.png'), cell_label.astype('uint16'))
        cv2.imwrite(os.path.join(sf, cell_output_name[0].split('_')[0] + '_combined_labels_color.png'), bytescale(image_label_overlay).astype('uint8'))
        cv2.imwrite(os.path.join(sf, os.path.splitext(cell_output_name[0])[0] + '_skeleton.png'), bytescale(sk*255.0).astype('uint8'))



def download_example_nuclei_dataset(dataset_path):
    os.makedirs(os.path.abspath(dataset_path),exist_ok=True)
    example_nuclei_dataset_url = "https://dl.dropbox.com/s/u5ccevdxewpxdup/HeLa_DAPI_v2.zip"
    example_nuclei_dataset = os.path.join(dataset_path, "HeLa_DAPI_v2.zip")
    download_with_url(example_nuclei_dataset_url, example_nuclei_dataset, unzip=True)

def generate_regression_config(dataset_path, input_channel_files, target_channel_files, phase='train', folder='dpn_unet_f0'):
    config = get_default_config()
    config.phase = phase
    config.folder = folder
    config.dataset_path =  dataset_path
    config.outputs_dir = os.path.join(config.dataset_path, config.folder, 'outputs')
    config.weights_dir = os.path.join(config.dataset_path, config.folder, 'weights')
    config.logs_dir = os.path.join(config.dataset_path, config.folder, 'logs')
    config.nb_epoch = 100000
    config.batch_size = 1
    config.epoch_size = 1
    config.target_cols = 256
    config.target_rows = 256
    config.loss = {"type": "mse"}
    config.activation = 'linear'
    config.input_channel_files = input_channel_files
    config.target_channel_files = target_channel_files
    return config

def generate_segmentation_config(dataset_path, input_channel_files, target_channel_files, load_model_from=None, scale_factor=1.0, phase='train', folder='dpn_unet_f0', device='cpu'):
    config = get_default_config()
    config.phase = phase
    config.folder = folder
    config.dataset_path =  dataset_path
    config.outputs_dir = os.path.join(config.dataset_path, config.folder, 'outputs')
    config.weights_dir = os.path.join(config.dataset_path, config.folder, 'weights')
    config.logs_dir = os.path.join(config.dataset_path, config.folder, 'logs')
    config.nb_epoch = 100000
    config.batch_size = 1
    config.epoch_size = 1
    config.target_cols = 256
    config.target_rows = 256
    config.loss = {"type": "bce", "ce": 0.6, "dice_body": 0.2, "dice_border": 0.2}
    config.activation = 'softmax'
    config.input_channel_files = input_channel_files
    config.target_channel_files = target_channel_files
    config.scale_factor = scale_factor
    config.load_model_from = None
    config.device = device
    return config

def start_training(config):
    if config.load_model_from is not None and os.path.exists(config.load_model_from):
        try:
            model = torch.load(config.load_model_from, map_location=torch.device('cpu'))
            print('loaded model from ' + config.load_model_from)
        except:
            print('Model weight file does not exist or is not valid => Train from scratch')
            model = None
    else:
        model = None

    paths, fn_mapping  = load_path_mapping(config, True)

    print(f'model will be saved to {config.weights_dir}')
    print('start training ...')

    fn_mapping = {
        'masks': lambda name: ['{}/{}'.format(name, fn) for fn in config.target_channel_files],
        'images': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
        'labels': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
    }

    preprocessing = {
        'images': get_preprocessing(config.inputs_preprocessing),
        'masks': get_preprocessing(config.targets_preprocessing),
        'labels': get_preprocessing('identity')
    }

    #torch.backends.cudnn.benchmark = True

    im_type = BorderImageType if not config.activation == 'sigmoid' else SigmoidBorderImageType
    im_val_type = PaddedImageType if not config.activation == 'sigmoid' else PaddedSigmoidImageType

    assert config.phase == 'train'
    paths_train = {
        'masks': os.path.join(config.dataset_path, 'train'),
        'images': os.path.join(config.dataset_path, 'train'),
        'labels': os.path.join(config.dataset_path, 'train'),
    }

    valid_path = os.path.join(config.dataset_path, 'valid') if os.path.exists(os.path.join(config.dataset_path, 'valid')) else os.path.join(config.dataset_path, 'test')
    paths_valid = {
        'masks': valid_path,
        'images': valid_path,
        'labels': valid_path,
    }
    ds = CachingImageProvider(im_type, paths_train, fn_mapping, preprocessing=preprocessing, scale_factor=config.scale_factor)
    val_ds = CachingImageProvider(im_val_type, paths_valid, fn_mapping, preprocessing=preprocessing, scale_factor=config.scale_factor)
    train_idx, val_idx = list(range(len(ds))), list(range(len(val_ds)))
    assert len(train_idx)>0, "No data found for training."

    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.weights_dir, exist_ok=True)
    os.makedirs(config.outputs_dir, exist_ok=True)

    set_output_dir(config.outputs_dir)
    with open(os.path.join(config.weights_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f)
    
    if len(config.input_channel_files) > 3:
        input_channel_num_changed = True
    else:
        input_channel_num_changed = False
    
    if model is None and os.path.exists(os.path.join(config.weights_dir, "__checkpoint__.pth")):
        print('Will try to resume from the checkpoint: ' + os.path.join(config.weights_dir, "__checkpoint__.pth") )
        checkpoint = "__checkpoint__.pth"
    else:
        checkpoint = None

    if model is None:
        model = unet.DPNUnet(target_channel_num=config.target_channel_num, input_channel_num=config.input_channel_num)
    estimator = Estimator(model, optim.Adam, config.weights_dir,
                        config=config, input_channel_num_changed=input_channel_num_changed, final_changed=False, device=config.device)


    estimator.lr_scheduler = ExponentialLR(estimator.optimizer, config.lr_gamma) # LRStepScheduler(estimator.optimizer, config.lr_steps)
    callbacks = [
        ModelSaver(1, ("dpn_unet_best.pth"), best_only=True),
        ModelSaver(1, ("dpn_unet_last.pth"), best_only=False),
        CheckpointSaver(1, ("__checkpoint__.pth")),
        # LRDropCheckpointSaver(("fold"+str(fold)+"_checkpoint_e{epoch}.pth")),
        ModelFreezer(),
        PredictionSaver(config.outputs_dir, scale_target=True, report_interval=config.save_interval),
        # EarlyStopper(10),
        TensorBoard(config.logs_dir)
    ]

    hard_neg_miner = None # HardNegativeMiner(rate=10)



    trainer = PytorchTrain(estimator,
                        callbacks=callbacks,
                        checkpoint=checkpoint,
                        hard_negative_miner=hard_neg_miner)
    
    if config.training_augmentations == 'binarized':
        training_augmenter = aug_binarize(prob=1)
    elif config.training_augmentations == 'victor':
        training_augmenter = aug_victor(.97)
    else:
        raise NotImplementedError
    train_loader = PytorchDataLoader(TrainDataset(ds, train_idx, config, transforms=training_augmenter),
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=config.num_workers,
                                    pin_memory=True)
    val_loader = PytorchDataLoader(ValDataset(val_ds, val_idx, config, transforms=None),
                                batch_size=1,
                                shuffle=False,
                                drop_last=False,
                                num_workers=config.num_workers,
                                pin_memory=True)

    trainer.fit(train_loader, val_loader, config.nb_epoch)

def start_segmentation_inference(config, apply_postprocessing=True):
    import cv2
    if config.load_model_from is not None:
        model = torch.load(config.load_model_from, map_location=torch.device('cpu'))
        print('loaded model from ' + config.load_model_from)
        resume = True
    else:
        model = None
        resume = False

    _, fn_mapping  = load_path_mapping(config, True)
    print(f'model will be saved to {config.weights_dir}')

    print('start inference ...')
     # if os.name == 'nt' else 4

    fn_mapping = {
        'masks': lambda name: ['{}/{}'.format(name, fn) for fn in config.target_channel_files],
        'images': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
        'labels': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
    }

    preprocessing = {
        'images': get_preprocessing(config.inputs_preprocessing),
        'masks': get_preprocessing(config.targets_preprocessing),
        'labels': get_preprocessing('identity')
    }

    im_val_type = PaddedImageType if not config.activation == 'sigmoid' else PaddedSigmoidImageType

    paths_test = {
        'masks': os.path.join(config.dataset_path, 'test'),
        'images': os.path.join(config.dataset_path, 'test'),
        'labels': os.path.join(config.dataset_path, 'test'),
    }

    ds = InFolderImageProvider(im_val_type, paths_test, fn_mapping, preprocessing=preprocessing, scale_factor=config.scale_factor)
    test_idx = list(range(len(ds)))
    test_dataset = SequentialDataset(ds, test_idx, stage='test', config=config, transforms=None)
    test_dl = PytorchDataLoader(test_dataset, batch_size=config.predict_batch_size, num_workers=config.num_workers, drop_last=False)

    model = read_model(config.load_model_from, config.device)
    pbar = tqdm.tqdm(test_dl, total=len(test_dl))
    for data in pbar:
        samples = data['image']
        samples.to(config.device)
        predicted = predict8tta(model, samples, config.activation)
        input_shape = data['image_shape'][:-1]
        input_shape = [int(float(input_shape[0])*config.scale_factor), int(float(input_shape[1])*config.scale_factor)]
        names = data['image_name']
        for i in range(len(names)):
            prediction = cut_border(predicted[i,...], input_shape)
            prediction = np.squeeze(prediction)
            name = names[i]
            if prediction.shape[2] < 3:
                rows, cols = prediction.shape[:2]
                zeros = np.zeros((rows, cols), dtype=np.float32)
                prediction = np.dstack((prediction[...,0], prediction[...,1], zeros))
            else:
                prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

            name = os.path.split(name)[-1]
            outputs_file_name, outputs_file_extension = os.path.splitext(ds.fn_mapping['masks'](name)[0])
            outputs_file = os.path.join(config.outputs_dir, outputs_file_name + '_output' + outputs_file_extension)

            folder, _ = os.path.split(outputs_file)
            if not os.path.exists(folder):
                os.makedirs(folder)

            img = (prediction * [255,255,0]).astype(np.uint8)
            img = cv2.resize(img, (0, 0), fx=1/config.scale_factor, fy=1/config.scale_factor)
            cv2.imwrite(outputs_file, img)

            print('result saved to ' + outputs_file)
            if apply_postprocessing:
                out_img = imread(outputs_file)
                label_image = wsh(out_img[...,2] / 255., 0.3, 1 - out_img[...,1] / 255., out_img[...,2] / 255)
                propsa = regionprops(label_image)
                cell_count = len(propsa)

                base_path, _ = os.path.splitext(outputs_file)
                cv2.imwrite(base_path + '_labels.png', label_image.astype('uint16'))
                image_label_overlay = label2rgb(label_image, bg_label=0, bg_color=(0.8, 0.8, 0.8), colors=jet_colors)
                cv2.imwrite(base_path + '_color_labels.png', bytescale(image_label_overlay).astype('uint8'))

def start_nuclei_seeded_segementation(config, nuclei_model, nuclei_channel_filename, cell_model):
    cell_channels = config.input_channel_files
    nuclei_channels = [nuclei_channel_filename] * 3
    cell_output_name = config.target_channel_files
    nuclei_output_name = [os.path.splitext(cell_channels[2])[0] + '_border_mask.png']

    config.input_channel_files = nuclei_channels
    config.load_model_from = nuclei_model
    config.target_channel_files = nuclei_output_name
    start_segmentation_inference(config, apply_postprocessing=False)

    config.input_channel_files = cell_channels
    config.load_model_from = cell_model
    config.target_channel_files = cell_output_name
    start_segmentation_inference(config, apply_postprocessing=False)

    wsh_nuclei_seeded(config, nuclei_output_name, cell_output_name)    

def start_regression_inference(config):
    import cv2
    if config.load_model_from is not None:
        model = torch.load(config.load_model_from, map_location=torch.device('cpu'))
        print('loaded model from ' + config.load_model_from)
        resume = True
    else:
        model = None
        resume = False

    _, fn_mapping  = load_path_mapping(config, True)

    print('start inference ...')
     # if os.name == 'nt' else 4

    fn_mapping = {
        'masks': lambda name: ['{}/{}'.format(name, fn) for fn in config.target_channel_files],
        'images': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
        'labels': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
    }

    preprocessing = {
        'images': get_preprocessing(config.inputs_preprocessing),
        'masks': get_preprocessing(config.targets_preprocessing),
        'labels': get_preprocessing('identity')
    }

    im_val_type = PaddedImageType if not config.activation == 'sigmoid' else PaddedSigmoidImageType

    paths_test = {
        'masks': os.path.join(config.dataset_path, 'test'),
        'images': os.path.join(config.dataset_path, 'test'),
        'labels': os.path.join(config.dataset_path, 'test'),
    }
    ds = InFolderImageProvider(im_val_type, paths_test, fn_mapping, preprocessing=preprocessing, scale_factor=config.scale_factor)
    test_idx = list(range(len(ds)))
    test_dataset = SequentialDataset(ds, test_idx, stage='test', config=config, transforms=None)
    test_dl = PytorchDataLoader(test_dataset, batch_size=config.predict_batch_size, num_workers=config.num_workers, drop_last=False)

    model = read_model(config.load_model_from, config.device)
    pbar = tqdm.tqdm(test_dl, total=len(test_dl))
    for data in pbar:
        samples = data['image']
        samples.to(config.device)
        input_shape = data['image_shape'][:-1]
        input_shape = [int(float(input_shape[0])*config.scale_factor), int(float(input_shape[1])*config.scale_factor)]
        predicted = predict8tta(model, samples, config.activation)
        names = data['image_name']
        for i in range(len(names)):
            name = names[i]
            prediction = cut_border(predicted[i,...], input_shape)
            img = (np.clip(prediction * 65535, 0, 65535)).astype(np.uint16)
            name = os.path.split(name)[-1]
            outputs_file_name, outputs_file_extension = os.path.splitext(ds.fn_mapping['masks'](name)[0])
            outputs_file = os.path.join(config.outputs_dir, outputs_file_name + '_output' + outputs_file_extension)
            folder, _ = os.path.split(outputs_file)
            if not os.path.exists(folder):
                os.makedirs(folder)
            print('saving to ',  outputs_file)
            if config.scale_factor != 1.0:
                img = cv2.resize(img, (0, 0), fx=1/config.scale_factor, fy=1/config.scale_factor)
            imwrite(outputs_file, img)