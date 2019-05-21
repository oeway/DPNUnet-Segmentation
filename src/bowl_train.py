import torch
import os
import numpy as np
import random

from utils import get_csv_folds, update_config, get_folds, cleanup_mac_hidden_files
from config import Config
from dataset.reading_image_provider import ReadingImageProvider, CachingImageProvider, InFolderImageProvider
from dataset.image_types import SigmoidBorderImageType, BorderImageType, PaddedImageType, PaddedSigmoidImageType
from pytorch_utils.concrete_eval import FullImageEvaluator
from augmentations.transforms import aug_victor
from pytorch_utils.train import train
from merge_preds import merge_files
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--training', action='store_true')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
    cfg['dataset_path'] = cfg['dataset_path'] + ('/train' if args.training else '/test')
config = Config(**cfg)

paths = {
    'masks': '',
    'images': '',
    'labels': '',
}

mask_name, mask_ext = os.path.splitext(config.mask_file_name)

fn_mapping = {
    'masks': lambda name: '{}/{}'.format(name, config.mask_file_name if args.training else mask_name+'_output'+mask_ext ),
    'images': lambda name: '{}/{}'.format(name, config.image_file_name),
    'labels': lambda name: name
}


if args.training:
    paths = {k: os.path.join(config.dataset_path, p) for k, p in paths.items()}
else:
    paths = {"images": config.dataset_path}

num_workers = 0 if os.name == 'nt' else 4

def train_bowl():
    global config
    torch.backends.cudnn.benchmark = True
    cleanup_mac_hidden_files(config.dataset_path)
    sample_count = len(os.listdir(config.dataset_path))
    idx = list(range(sample_count))
    random.seed(1)
    random.shuffle(idx)
    split = 0.95
    train_idx, val_idx = idx[:int(split*sample_count)], idx[int(split*sample_count):] 
    im_type = BorderImageType if not config.sigmoid else SigmoidBorderImageType
    im_val_type = PaddedImageType if not config.sigmoid else PaddedSigmoidImageType
    ds = CachingImageProvider(im_type, paths, fn_mapping)
    val_ds = CachingImageProvider(im_val_type, paths, fn_mapping)
    fold = args.fold
    train(ds, val_ds, fold, train_idx, val_idx, config, num_workers=num_workers, transforms=aug_victor(.97))


def eval_bowl():
    global config
    test = not args.training
    cleanup_mac_hidden_files(config.dataset_path)
    sample_count = len(os.listdir(config.dataset_path))
    val_indexes = list(range(sample_count))
    im_val_type = PaddedImageType if not config.sigmoid else PaddedSigmoidImageType
    im_prov_type = InFolderImageProvider if test else ReadingImageProvider
    ds = im_prov_type(im_val_type, paths, fn_mapping)
    keval = FullImageEvaluator(config, ds, test=test, flips=3, num_workers=num_workers, border=0)
    fold = args.fold
    keval.predict(fold, val_indexes)
    if test and args.fold is None:
        merge_files(keval.save_dir)

if __name__ == "__main__":
    train_bowl()

