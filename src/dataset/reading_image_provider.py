import os

from .abstract_image_provider import AbstractImageProvider
import numpy as np


class ReadingImageProvider(AbstractImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False, scale_factor=1.0):
        super(ReadingImageProvider, self).__init__(image_type, fn_mapping, has_alpha=has_alpha, scale_factor=scale_factor)
        self.im_names = os.listdir(paths['images'])
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        indexes = {os.path.splitext(name)[0]: idx for idx, name in enumerate(self.im_names)}
        ret = [indexes[name] for name in names if name in indexes]
        return ret

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.has_alpha, self.scale_factor)

    def __len__(self):
        return len(self.im_names)


class CachingImageProvider(ReadingImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False, scale_factor=1.0):
        super().__init__(image_type, paths, fn_mapping, image_suffix, has_alpha=has_alpha, scale_factor=scale_factor)
        self.cache = {}

    def __getitem__(self, item):
        if item not in self.cache:
            data = super().__getitem__(item)
            self.cache[item] = data
        return self.cache[item]

class InFolderImageProvider(ReadingImageProvider):
    def __init__(self, image_type, paths, fn_mapping=lambda name: name, image_suffix=None, has_alpha=False, scale_factor=1.0):
        super().__init__(image_type, paths, fn_mapping, image_suffix, has_alpha, scale_factor=scale_factor)

    def __getitem__(self, item):
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.has_alpha, scale_factor=scale_factor)
