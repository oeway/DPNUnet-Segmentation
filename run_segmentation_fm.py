# Specify data-set
dataset_path = './data/HeLa_DAPI_v2'
dataset_path = 'D:\\Documents\\Data\\collaborations\\rensen-hiv\\200127_hiv_gfp\\analysis\\S10_20X\\DPNUnet_segmentation\\prediction'

# Imports
import os
from utils import allocate_gpu, generate_geojson_masks, download_with_url
from helpers import wsh_nuclei_seeded
from distutils.dir_util import copy_tree
from configs import get_default_config
from helpers import start_training, start_segmentation_inference, start_nuclei_seeded_segementation, \
    download_example_nuclei_dataset
    
# CPU / GPU computing
try:
    allocate_gpu()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    print('Using device: ' + device)
except:
    print('No GPU available, will use CPU')
    device = 'cpu'


# Verify if data is organized correctly
if not (os.path.exists(os.path.join(dataset_path, 'test')) or os.path.exists(os.path.join(dataset_path, 'train'))):
    print('No data found. Downloading test data from server...')
    download_example_nuclei_dataset(dataset_path)
    generate_geojson_masks(dataset_path, border_detection_threshold=3)

# Specify model path
models_folder = os.path.abspath('../models')
models_folder = 'D:\\Documents\\Data\ImJoy\\DPNUnet\\models'

# Config
config = get_default_config()

config.phase = 'segment_nuclei'
config.input_channel_files = ['nuclei.png']
config.target_channel_files = ['nuclei_border_mask.png']
config.folder = 'HeLa_DAPI_DPNUnet'  # REALLY NEEDED???
config.dataset_path = dataset_path

config.outputs_dir = os.path.join(config.dataset_path, config.folder, 'outputs')
config.weights_dir = os.path.join(config.dataset_path, config.folder, 'weights')
config.logs_dir = os.path.join(config.dataset_path, config.folder, 'logs')

config.nb_epoch = 100000
config.batch_size = 4  # Decrease batch size in case memory runs out
config.epoch_size = 14
config.target_cols = 256
config.target_rows = 256
config.loss = {"type": "bce", "ce": 0.6, "dice_body": 0.2, "dice_border": 0.2}
config.activation = 'softmax'
config.input_channel_num = 3
config.target_channel_num = 3
config.scale_factor = 0.25
config.device = device
config.save_interval = 8
config.inputs_preprocessing = 'min-max'
config.targets_preprocessing = 'bit-depth'
config.num_workers = 0 # set to 0 in vscode debugger. FM: values > 0 trigger errors under windows

if config.phase == 'train':
    if not os.path.exists(os.path.join(config.weights_dir, '__checkpoint__.pth')) and config.load_model_from is None:
        nuclei_model = os.path.join(models_folder, "dpn_unet_cell_v1.pth.pth")
        if not os.path.exists(nuclei_model):
            os.makedirs(models_folder,exist_ok=True)
            nuclei_model = os.path.join(models_folder, "dpn_unet_cell_v1.pth.pth")
            print('Downloading nuclei segmentation model...')
            nuclei_model_url = "https://kth.box.com/shared/static/l8z58wxkww9nn9syx9z90sclaga01mad.pth"
            download_with_url(nuclei_model_url, nuclei_model)
        config.load_model_from = nuclei_model
    start_training(config)
    
elif config.phase == 'segment_nuclei':
    
    config.outputs_dir = os.path.join(config.dataset_path, 'test')
    model = os.path.join(models_folder, "dpn_unet_nuclei_v1.pth")
    
    # Check if model is present
    if not os.path.exists(model):
        os.makedirs(models_folder,exist_ok=True)
        print('Downloading nuclei segmentation model...')
        model_url = "https://kth.box.com/shared/static/l8z58wxkww9nn9syx9z90sclaga01mad.pth"
        download_with_url(model_url, model)
    
    # Start prediction
    config.load_model_from = model
    config.input_channel_files = ['nuclei.png', 'nuclei.png', 'nuclei.png']
    config.target_channel_files = ['nuclei_border_mask.png']
    config.outputs_dir = os.path.join(config.dataset_path, 'test')
    start_segmentation_inference(config, apply_postprocessing=True)
        
    
elif config.phase == 'segment_with_nuclei':
    config.outputs_dir = os.path.join(config.dataset_path, 'test')
    models_folder = os.path.abspath('../models')
    config.scale_factor = 0.25

    nuclei_model = os.path.join(models_folder, "dpn_unet_nuclei.pth")
    if not os.path.exists(nuclei_model):
        os.makedirs(models_folder,exist_ok=True)
        nuclei_model = os.path.join(models_folder, "dpn_unet_nuclei.pth")
        print('Downloading nuclei segmentation model...')
        nuclei_model_url = "https://kth.box.com/shared/static/l8z58wxkww9nn9syx9z90sclaga01mad.pth"
        download_with_url(nuclei_model_url, nuclei_model)

    cell_model = os.path.join(models_folder, "dpn_unet_cell.pth")
    if not os.path.exists(cell_model):
        os.makedirs(models_folder,exist_ok=True)
        cell_model = os.path.join(models_folder, "dpn_unet_cell.pth")
        print('Downloading cell segmentation model...')
        cell_model_url = "https://kth.box.com/shared/static/he8kbtpqdzm9xiznaospm15w4oqxp40f.pth"
        download_with_url(cell_model_url, cell_model)

    config.input_channel_files = ['microtubules.png', None, 'nuclei.png']
    nuclei_channel_filename = 'nuclei.png'

    start_nuclei_seeded_segementation(config, nuclei_model, nuclei_channel_filename, cell_model)

else:
    raise NotImplementedError
