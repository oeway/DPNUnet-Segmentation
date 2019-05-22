import os

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np

from .eval import Evaluator


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        if 'step_callback' in kwargs:
            self.step_callback = kwargs['step_callback']
            del kwargs['step_callback']
        else:
            self.step_callback = None
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix=""):
        if self.test:
            path = os.path.join(self.config.dataset_path, self.ds.fn_mapping['images'](name))
        else:
            path = os.path.join(self.config.dataset_path, 'images_all', name)
        rows, cols = cv2.imread(path, 0).shape[:2]
        prediction = prediction[0:rows, 0:cols,...]
        if prediction.shape[2] < 3:
            zeros = np.zeros((rows, cols), dtype=np.float32)
            prediction = np.dstack((prediction[...,0], prediction[...,1], zeros))
        else:
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        if self.test:
            name = os.path.split(name)[-1]
        
        save_path = os.path.join(self.save_dir, self.ds.fn_mapping['masks'](name)) # os.path.join(self.save_dir + prefix, self.ds.fn_mapping['masks'](name))

        do_save = True
        if self.step_callback is not None:
            skip_save = self.step_callback({'prediction': prediction, 'save_path': save_path, 'name': name})
            if skip_save:
                do_save = False

        if do_save:
            print('saving to ',  save_path)
            folder, name = os.path.split(save_path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(save_path, (prediction * [255,255,0]).astype(np.uint8))