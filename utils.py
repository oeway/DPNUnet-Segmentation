import cv2
import os
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, remove_small_holes, skeletonize, binary_closing,\
    binary_dilation, binary_erosion
from skimage import measure, exposure, segmentation
from imgseg.geojson_utils import gen_mask_from_geojson

from imageio import imread
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import random
from matplotlib import cm
from scipy.ndimage.morphology import binary_fill_holes
import urllib.request
import zipfile


def bytescale(arr):
    ma = arr.max()
    mi = arr.min()
    if ma == mi:
        return arr*0
    return (arr- mi)/(ma - mi)*255


def download_with_url(url_string, file_path, unzip=False):
    with urllib.request.urlopen(url_string) as response, open(file_path, 'wb') as out_file:
        data = response.read() # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))
            
def fill_holes(image):
        """fill_holes for labelled image, with each object has a unique number"""
        boundaries = segmentation.find_boundaries(image)
        image = np.multiply(image, np.invert(boundaries))
        image = binary_fill_holes(image > 0)
        image = ndi.label(image)[0]
        return image


def wsh(mask_img, threshold, border_img, seeds, threshold_adjustment=0.35, small_object_size_cutoff=10):
    img_copy = np.copy(mask_img)
    m = seeds * border_img# * dt
    img_copy[m <= threshold + threshold_adjustment] = 0
    img_copy[m > threshold + threshold_adjustment] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = remove_small_objects(img_copy, small_object_size_cutoff).astype(np.uint8)

    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_holes(mask_img, 1000)
    mask_img = remove_small_objects(mask_img, 8).astype(np.uint8)
    markers = ndi.label(img_copy, output=np.uint32)[0]
    labeled_array = watershed(mask_img, markers, mask=mask_img, watershed_line=True)
    return labeled_array

def merge_files(root):
    res_path = os.path.join('..', '..', 'predictions', os.path.split(root)[-1] + '_test')
    os.makedirs(res_path, exist_ok=True)
    prob_files = {f for f in os.listdir(root) if os.path.splitext(f)[1] in ['.png']}
    unfolded = {f[6:] for f in prob_files if f.startswith('fold')}
    if not unfolded:
        unfolded = prob_files

    for prob_file in tqdm.tqdm(unfolded):
        probs = []
        for fold in range(4):
            prob = os.path.join(root, 'fold{}_'.format(fold) + prob_file)
            prob_arr = cv2.imread(prob, cv2.IMREAD_UNCHANGED)
            probs.append(prob_arr)
        prob_arr = np.mean(probs, axis=0)

        res_path_geo = os.path.join(res_path, prob_file)
        cv2.imwrite(res_path_geo, prob_arr)

def cut_border(image, target_shape):
        assert len(target_shape) == 2
        return image if not target_shape else image[32: 32+target_shape[0], 32:32+target_shape[1], ... ]

def read_model(model_path, device):
    import torch
    from torch import nn
    if not os.path.exists(model_path):
        raise Exception('Model not exists.')
    if device == 'cpu':
        model = torch.load(model_path, map_location=torch.device('cpu')).to(device)
    else:
        model = nn.DataParallel(torch.load(model_path, map_location=torch.device('cpu'))).to(device)
    model.eval()
    return model


def load_path_mapping(config, training):
    paths = {
        'masks': '',
        'images': '',
        'labels': '',
    }
#    mask_name, mask_ext = os.path.splitext(config.target_channel_files)
    if training:
        fn_mapping = {
            'masks': lambda name: ['{}/{}'.format(name, fn) for fn in config.target_channel_files],
            'images': lambda name: ['{}/{}'.format(name, fn) for fn in config.input_channel_files],
            'labels': lambda name: name
        }
    else:
        fn_mapping = {
            'masks': lambda name: ['{}/{}'.format(name, fn) for fn in config.target_channel_files],
#            'masks': lambda name: '{}/{}'.format(name, mask_name+'_output'+mask_ext ),
            'images': lambda name: ['{}/{}'.format(name,  fn) for fn in config.input_channel_files],
            'labels': lambda name: name
        }
    if training:
        paths = {k: os.path.join(config.dataset_path, p) for k, p in paths.items()}
    else:
        paths = {"images": config.dataset_path}
    return paths, fn_mapping


def _gen_masks(datasets_dir, file_ids, border_detection_threshold=6):

    for i, file_id in enumerate(file_ids):
        print('Processing ' + file_id)
        file_path = os.path.join(datasets_dir, file_id, "annotation.json")
        try:
            gen_mask_from_geojson([file_path], masks_to_create_value=["border_mask"], border_detection_threshold=border_detection_threshold)
        except:
            print("generate mask error:", os.path.join(datasets_dir, "train", file_id))

def generate_geojson_masks(datasets_dir, border_detection_threshold=6):
    print("datasets_dir:", datasets_dir)
    file_ids = os.listdir(os.path.join(datasets_dir, "train"))
    _gen_masks(os.path.join(datasets_dir, "train"), file_ids, border_detection_threshold=border_detection_threshold)
    if os.path.exists(os.path.join(datasets_dir, "valid")):
        file_ids = os.listdir(os.path.join(datasets_dir, "valid"))
        _gen_masks(os.path.join(datasets_dir, "valid"), file_ids, border_detection_threshold=border_detection_threshold)
    if os.path.exists(os.path.join(datasets_dir, "test")):
        file_ids = os.listdir(os.path.join(datasets_dir, "test"))
        _gen_masks(os.path.join(datasets_dir, "test"), file_ids, border_detection_threshold=border_detection_threshold)
    print('Finished, generated masks saved to ' + datasets_dir)


def allocate_gpu(num=1, gpu_ids=None):
    if os.environ.get("NVIDIA_VISIBLE_DEVICES"):
        DEVICE_ID = os.environ.get("NVIDIA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
        print(f'GPU id is set to : {DEVICE_ID}')
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        DEVICE_ID = os.environ.get("CUDA_VISIBLE_DEVICES")
        print(f'GPU id is set to : {DEVICE_ID}')
        return

    import GPUtil
    if gpu_ids is None:
        # Get the first available GPU
        DEVICE_ID_LIST = GPUtil.getAvailable(order = 'first', limit = num, maxLoad = 0.8, maxMemory = 0.8)
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f'Available GPUs: {DEVICE_ID_LIST}')
        if len(DEVICE_ID_LIST)<= 0:
            print('No GPU available')
        print(f'Set GPU id to : {DEVICE_ID_LIST}')
        # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, DEVICE_ID_LIST))
    else:
        GPUs = [g.id for g in GPUtil.getGPUs()]
        for i in gpu_ids:
            if i not in GPUs:
                raise Exception(f'Invalid GPU Id:{i}')
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))

def bit_depth_normalization(img):
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        return (img / 65535.0 * 255.0).astype('uint8')
    elif img.dtype == np.float32:
        return (img * 255.0).astype('uint8')
    else:
        raise NotImplementedError


def get_preprocessing(type):
    if type == 'identity':
        return lambda x: x
    elif type == 'min-max':
        return lambda x: bytescale(x).astype('uint8')
    elif type == 'bit-depth':
        return bit_depth_normalization
    elif type == 'equalize-adapthist':
        return lambda x: exposure.equalize_adapthist(x, clip_limit=0.01).astype('uint8')
    else:
        raise NotImplementedError
