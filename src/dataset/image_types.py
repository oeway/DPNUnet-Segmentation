import os
import numpy as np
import cv2
from scipy.misc import imread
from dataset.raw_image import RawImageType

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

