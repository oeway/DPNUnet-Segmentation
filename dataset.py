from augmentations.transforms import ToTensor
import numpy as np
import os
import random
from imageio import imread, imwrite
from typing import Type, Dict, AnyStr, Callable

import os
import sys


import random
from matplotlib import cm
jet_colors = [cm.jet(i)[:3] for i in range(256)]
random.shuffle(jet_colors)


import os
import numpy as np

import numpy as np
import pandas as pd

import cv2
import os
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from skimage import measure

import json


import base64
from shutil import copyfile

import argparse

import os
from collections import defaultdict
from typing import Type, Dict, AnyStr, Callable

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from typing import Type


import torch.nn.functional as F
from augmentations.tta import transforms as TTA

output_dir = None
def set_output_dir(d):
    global output_dir
    output_dir = d
    os.makedirs(os.path.join(output_dir, '_data'), exist_ok=True)

class AbstractImageType:
    def __init__(self, paths, fn, fn_mapping, preprocessing, has_alpha=False, scale_factor=1.0):
        self.paths = paths
        self.fn = fn
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping
        self.preprocessing = preprocessing
        self.scale_factor = scale_factor
        self.cache = {}

    @property
    def image(self):
        if 'image' not in self.cache:
            im = self.read_image()
            self.cache['image'] = im
            if output_dir is not None:
                imwrite(os.path.join(output_dir, '_data',  self.fn+"_image.png"), im)
        return np.copy(self.cache['image'])

    @property
    def mask(self):
        if 'mask' not in self.cache:
            im = self.read_mask()
            self.cache['mask'] = im
            if output_dir is not None:
                imwrite(os.path.join(output_dir, '_data',  self.fn+"_mask.png"), im)
        return np.copy(self.cache['mask'])

    @property
    def alpha(self):
        if not self.has_alpha:
            raise AlphaNotAvailableException
        if 'alpha' not in self.cache:
            im = self.read_alpha()
            if output_dir is not None:
                imwrite(os.path.join(output_dir, '_data',  self.fn+"_alpha.png"), im)
            self.cache['alpha'] = im
        return self.cache['alpha']

    @property
    def label(self):
        if 'label' not in self.cache:
            im = self.read_label()
            if output_dir is not None:
                imwrite(os.path.join(output_dir, '_data',  self.fn+"_label.png"), im)
            self.cache['label']  = im
        return np.copy(self.cache['label'])


    def read_alpha(self):
        raise NotImplementedError

    def read_image(self):
        raise NotImplementedError

    def read_mask(self):
        raise NotImplementedError

    def read_label(self):
        raise NotImplementedError

    def reflect_border(self, image, b=12):
        return cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_REFLECT)

    def pad_image(self, image, rows, cols):
        channels = image.shape[2] if len(image.shape) > 2 else None
        if image.shape[:2] != (rows, cols):
            empty_x = np.zeros((rows, cols, channels), dtype=image.dtype) if channels else np.zeros((rows, cols), dtype=image.dtype)
            empty_x[0:image.shape[0],0:image.shape[1],...] = image
            image = empty_x
        return image

    def finalyze(self, image):
        return self.reflect_border(image)


class RawImageType(AbstractImageType):
    def __init__(self, paths, fn, fn_mapping, preprocessing, has_alpha, scale_factor=1.0):
        super().__init__(paths, fn, fn_mapping, preprocessing, has_alpha, scale_factor=scale_factor)
        self.im = self.load_image_files('images')
        self.image_shape = self.im.shape
    
    def load_image_files(self, img_type):
        rgb_img = []
        for fn in self.fn_mapping[img_type](self.fn):
            if fn.endswith('None'):
                rgb_img.append(None)
            else:
                fpath = os.path.join(self.paths[img_type], fn)
                if os.path.exists(fpath):
                    img = imread(fpath)
                    if self.preprocessing is not None:
                        img = self.preprocessing[img_type](img)
                    if len(img.shape) == 2:
                        rgb_img.append(img)
                    else:
                        for i in range(img.shape[2]):
                            if i >= 3:
                                break
                            rgb_img.append(img[:, :, i])
                else:
                    rgb_img.append(None)

        assert len(rgb_img) ==1 or len(rgb_img) == 3 or len(rgb_img) == 4, f'Invalid image channel number: {len(rgb_img)}'
        ref_channel = None
        for i in range(len(rgb_img)):
            if rgb_img[i] is not None:
                ref_channel = rgb_img[i]
        assert ref_channel is not None
        for i in range(len(rgb_img)):
            if rgb_img[i] is None:
                rgb_img[i] = np.zeros_like(ref_channel)
        im = np.dstack(rgb_img).astype(np.uint8)
        return im

    def read_image(self):
        im = self.im[...,:-1] if self.has_alpha else self.im
        if self.scale_factor != 1.0:
            im = cv2.resize(im, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if im.ndim == 2:
            im = im[:,:,np.newaxis]
        return self.finalyze(im)

    def read_mask(self):
        mask = self.load_image_files('masks')
        if self.scale_factor != 1.0:
            mask = cv2.resize(mask, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if mask.ndim == 2:
            mask = mask[:,:,np.newaxis]
        return self.finalyze(mask)

    def read_alpha(self):
        im = self.im[...,-1]
        if self.scale_factor != 1.0:
            im = cv2.resize(im, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        return self.finalyze(im)

    def read_label(self):
        label = mask = self.load_image_files('labels')
        if self.scale_factor != 1.0:
            label = cv2.resize(label, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if label.ndim == 2:
            label = label[:,:,np.newaxis]
        return self.finalyze(label)

    def finalyze(self, data):
        return data

class MinSizeImageType(RawImageType):
    def finalyze(self, data):
        rows, cols = data.shape[:2]
        nrows = (256 - rows) if rows < 256 else 0
        ncols = (256 - cols) if cols < 256 else 0
        if nrows > 0 or ncols > 0:
            return cv2.copyMakeBorder(data, 0, nrows, 0, ncols, cv2.BORDER_CONSTANT)
        return data

class BorderImageType(MinSizeImageType):
    def read_mask(self):
        mask = self.load_image_files('masks')
        if self.scale_factor != 1.0:
            mask = cv2.resize(mask, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if mask.ndim == 2:
            mask = mask[:,:,np.newaxis]
        # msk[..., 2] = (msk[..., 2] > 127)
        # msk[..., 1] = (msk[..., 1] > 127) * (msk[..., 2] == 0)
        # msk[..., 0] = (msk[..., 1] == 0) * (msk[..., 2] == 0)
        return self.finalyze(mask)

class SigmoidBorderImageType(MinSizeImageType):
    def read_mask(self):
        mask = self.load_image_files('masks')
        if self.scale_factor != 1.0:
            mask = cv2.resize(mask, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if mask.ndim == 2:
            mask = mask[:,:,np.newaxis]
        label = self.read_label()
        fin = self.finalyze(mask)
        data = np.dstack((fin[...,2], fin[...,1], (label > 0).astype(np.uint8) * 255))
        return data

class PaddedImageType(BorderImageType):
    def finalyze(self, data):
        rows, cols = data.shape[:2]
        return cv2.copyMakeBorder(data, 32, (32-rows%32), 32, (32-cols%32), cv2.BORDER_REFLECT)

class PaddedSigmoidImageType(SigmoidBorderImageType):
    def finalyze(self, data):
        rows, cols = data.shape[:2]
        return cv2.copyMakeBorder(data, 32, (32-rows%32), 32, (32-cols%32), cv2.BORDER_REFLECT)

class AbstractImageProvider:
    def __init__(self, image_type: Type[AbstractImageType], fn_mapping: Dict[AnyStr, Callable], preprocessing: Dict[AnyStr, Callable], has_alpha=False, scale_factor=1.0):
        self.image_type = image_type
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping
        self.preprocessing = preprocessing
        self.scale_factor = scale_factor

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ReadingImageProvider(AbstractImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, preprocessing=None, image_suffix=None, has_alpha=False, scale_factor=1.0):
        super(ReadingImageProvider, self).__init__(image_type, fn_mapping, preprocessing, has_alpha=has_alpha, scale_factor=scale_factor)
        self.im_names = []
        for d in os.listdir(paths['images']):
            if not d.startswith('.'):
                skip = False
                for fn in self.fn_mapping['images'](d):
                    if not os.path.exists(os.path.join(paths['images'], fn)) and not fn.endswith('None'):
                        skip = True
                        print('Warning: skipping incompelete sample: ' + fn)
                        break
                if not skip:
                    self.im_names.append(d)
        print(f'{len(self.im_names)} samples found.')
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        indexes = {os.path.splitext(name)[0]: idx for idx, name in enumerate(self.im_names)}
        ret = [indexes[name] for name in names if name in indexes]
        return ret

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.preprocessing, self.has_alpha, self.scale_factor)

    def __len__(self):
        return len(self.im_names)

class CachingImageProvider(ReadingImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, preprocessing=None, image_suffix=None, has_alpha=False, scale_factor=1.0):
        super().__init__(image_type, paths, fn_mapping, preprocessing, image_suffix, has_alpha=has_alpha, scale_factor=scale_factor)
        self.cache = {}

    def __getitem__(self, item):
        if item not in self.cache:
            data = super().__getitem__(item)
            self.cache[item] = data
        return self.cache[item]

class InFolderImageProvider(ReadingImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, preprocessing=None, image_suffix=None, has_alpha=False, scale_factor=1.0):
        super().__init__(image_type, paths, fn_mapping, preprocessing, image_suffix, has_alpha, scale_factor=scale_factor)

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.preprocessing, self.has_alpha, scale_factor=self.scale_factor)

class ImageCropper:
    def __init__(self, img_rows, img_cols, target_rows, target_cols, pad):
        self.image_rows = img_rows
        self.image_cols = img_cols
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.pad = pad
        self.use_crop = (img_rows != target_rows) or (img_cols != target_cols)
        self.starts_y = self.sequential_starts(axis=0) if self.use_crop else [0]
        self.starts_x = self.sequential_starts(axis=1) if self.use_crop else [0]
        self.positions = [(x, y) for x in self.starts_x for y in self.starts_y]
        assert target_rows <=  img_rows and target_cols <= img_cols, f'Target size ({target_cols}, {target_rows}) must be smaller than the image size ({img_cols}, {img_rows})'
        # self.lock = threading.Lock()

    def random_crop_coords(self):
        x = random.randint(0, self.image_cols - self.target_cols)
        y = random.randint(0, self.image_rows - self.target_rows)
        return x, y

    def crop_image(self, image, x, y):
        self.use_crop = ((image.shape[0]) != self.target_rows) or ((image.shape[1]) != self.target_cols)
        return image[y: y+self.target_rows, x: x+self.target_cols,...] if self.use_crop else image

    def sequential_crops(self, img):
        for startx in self.starts_x:
            for starty in self.starts_y:
                yield self.crop_image(img, startx, starty)

    def sequential_starts(self, axis=0):
        big_segment = self.image_cols if axis else self.image_rows
        small_segment = self.target_cols if axis else self.target_rows
        if big_segment == small_segment:
            return [0]
        steps = np.ceil((big_segment - self.pad) / (small_segment - self.pad)) # how many small segments in big segment
        if steps == 1:
            return [0]
        new_pad = int(np.floor((small_segment * steps - big_segment) / (steps - 1))) # recalculate pad
        starts = [i for i in range(0, big_segment - small_segment, small_segment - new_pad)]
        starts.append(big_segment - small_segment)
        return starts

class BasicTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **kwargs):
        if random.random() < self.prob:
            params = self.get_params()
            return {k: self.apply(a, **params) if k in self.targets else a for k, a in kwargs.items()}
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError


class Compose:
    def __init__(self, transforms, prob=1.):
        self.transforms = [t for t in transforms if t is not None]
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:
            for t in self.transforms:
                data = t(**data)
        return data

class Dataset:
    def __init__(self, image_provider: AbstractImageProvider, image_indexes, config, stage='train', transforms=None):
        self.pad = 0 if stage=='train' else config.test_pad
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) else image_indexes.tolist()
        if stage != 'train' and len(self.image_indexes) % 2: #todo bugreport it
            self.image_indexes += [self.image_indexes[-1]]
        self.stage = stage
        self.keys = {'image', 'image_name', 'image_shape'}
        self.config = config
        normalize = {'mean': [124 / 255, 117 / 255, 104 / 255],
                     'std': [1 / (.0167 * 255)] * 3}
        self.transforms = Compose([transforms, ToTensor(config.target_channel_num, config.activation, normalize)])
        self.croppers = {}

    def __getitem__(self, item):
        raise NotImplementedError

    def get_cropper(self, image_id, val=False):
        #todo maybe cache croppers for different sizes too speedup if it's slow part?
        if image_id not in self.croppers:
            image = self.image_provider[image_id].image
            rows, cols = image.shape[:2]
            if self.config.ignore_target_size and val:
                assert self.config.predict_batch_size == 1
                target_rows, target_cols = rows, cols
            else:
                target_rows, target_cols = self.config.target_rows, self.config.target_cols
            cropper = ImageCropper(rows, cols,
                                   target_rows, target_cols,
                                   self.pad)
            self.croppers[image_id] = cropper
        return self.croppers[image_id]
      

class SequentialDataset(Dataset):
    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=None):
        super(SequentialDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'geometry'})

    def init_good_tiles(self):
        self.good_tiles = []
        for im_idx in self.image_indexes:
            cropper = self.get_cropper(im_idx, val=True)
            positions = cropper.positions
            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))
        
    def prepare_image(self, item, cropper, sx, sy):
        im = cropper.crop_image(item.image, sx, sy)
        rows, cols = item.image.shape[:2]
        geometry = {'rows': rows, 'cols': cols, 'sx': sx, 'sy': sy}
        data = {'image': im, 'image_name': item.fn, 'geometry': geometry, 'image_shape': item.image_shape}
        return data

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)
        return self.transforms(**data)

    def __len__(self):
        return len(self.good_tiles)


    
class TrainDataset(Dataset):
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=None, partly_sequential=False):
        super(TrainDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')
        self.partly_sequential = partly_sequential
        self.inner_idx = 9
        self.idx = 0
        masks = []
        labels = []
        # for im_idx in self.image_indexes:
        #     item = self.image_provider[im_idx]
        #     masks.append(item.mask)
        #     labels.append(item.label)
        # self.dv_cropper = DVCropper(masks, labels, config.target_rows, config.target_cols)


    def __getitem__(self, idx):
        if self.partly_sequential:
            #todo rewrite somehow better
            if self.inner_idx > 8:
                self.idx = idx
                self.inner_idx = 0
            self.inner_idx += 1
            im_idx = self.image_indexes[self.idx % len(self.image_indexes)]
        else:
            im_idx = self.image_indexes[idx % len(self.image_indexes)]

        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        sx, sy = cropper.random_crop_coords()
        if cropper.use_crop and self.image_provider.has_alpha:
            for i in range(10):
                alpha = cropper.crop_image(item.alpha, sx, sy)
                if np.mean(alpha) > 5:
                    break
                sx, sy = cropper.random_crop_coords()
            else:
                return self.__getitem__(random.randint(0, len(self.image_indexes)))

        im = cropper.crop_image(item.image, sx, sy)
        mask = cropper.crop_image(item.mask, sx, sy)
        # im, mask, lbl = item.image, item.mask, item.label
        # im, mask = self.dv_cropper.strange_method(idx % len(self.image_indexes), im, mask, lbl, sx, sy)
        data = {'image': im, 'mask': mask, 'image_name': item.fn, 'image_shape': item.image_shape}
        return self.transforms(**data)

    def __len__(self):
        return len(self.image_indexes) * max(self.config.epoch_size, 1) # epoch size is len images

      
class ValDataset(SequentialDataset):
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=None):
        super(ValDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx):
        im_idx, sx, sy = self.good_tiles[idx]
        
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)
        
        mask = cropper.crop_image(item.mask, sx, sy)
        data.update({'mask': mask})

        d = self.transforms(**data)
        return d
