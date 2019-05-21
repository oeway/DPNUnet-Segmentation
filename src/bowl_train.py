import torch
import os
import cv2
from scipy.misc import imread
import numpy as np
import random

from utils import get_csv_folds, update_config, get_folds, cleanup_mac_hidden_files
from config import Config
from dataset.reading_image_provider import ReadingImageProvider, CachingImageProvider, InFolderImageProvider
from dataset.raw_image import RawImageType
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

name, ext = os.path.splitext(config.mask_file_name)

fn_mapping = {
    'masks': lambda name: '{}/{}'.format(name, config.mask_file_name if args.training else name+'_output'+ext ),
    'images': lambda name: '{}/{}'.format(name, config.image_file_name),
    'labels': lambda name: name
}


if args.training:
    paths = {k: os.path.join(config.dataset_path, p) for k, p in paths.items()}
else:
    paths = {"images": config.dataset_path}
num_workers = 0 if os.name == 'nt' else 4

class MinSizeImageType(RawImageType):
    def finalyze(self, data):
        rows, cols = data.shape[:2]
        nrows = (256 - rows) if rows < 256 else 0
        ncols = (256 - cols) if cols < 256 else 0
        if nrows > 0 or ncols > 0:
            return cv2.copyMakeBorder(data, 0, nrows, 0, ncols, cv2.BORDER_CONSTANT)
        return data


class SigmoidBorderImageType(MinSizeImageType):
    def read_mask(self):
        path = os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn))
        mask = imread(path, mode='RGB')
        label = self.read_label()
        fin = self.finalyze(mask)
        data = np.dstack((fin[...,2], fin[...,1], (label > 0).astype(np.uint8) * 255))
        return data

class BorderImageType(MinSizeImageType):
    def read_mask(self):
        path = os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn))
        msk = imread(path, mode='RGB')
        msk[..., 2] = (msk[..., 2] > 127)
        msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 2] == 0)
        msk[..., 0] = (msk[..., 1] == 0) * (msk[..., 2] == 0)
        return self.finalyze(msk.astype(np.uint8) * 255)


class PaddedImageType(BorderImageType):
    def finalyze(self, data):
        rows, cols = data.shape[:2]
        return cv2.copyMakeBorder(data, 0, (32-rows%32), 0, (32-cols%32), cv2.BORDER_REFLECT)

class PaddedSigmoidImageType(SigmoidBorderImageType):
    def finalyze(self, data):
        rows, cols = data.shape[:2]
        return cv2.copyMakeBorder(data, 0, (32-rows%32), 0, (32-cols%32), cv2.BORDER_REFLECT)


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

