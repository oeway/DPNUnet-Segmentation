import os
import sys
import matplotlib.pyplot as plt


import os
import sys
import matplotlib.pyplot as plt

import os
import numpy as np

import numpy as np
import pandas as pd

import os
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from skimage import measure
from callbacks import Callbacks
from losses import dice_loss, dice_round, ssim


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

import tqdm
from typing import Type

from scipy.stats import pearsonr

import torch.nn.functional as F
from augmentations.tta import transforms as TTA
from utils import bytescale


class Estimator:
    def __init__(self, model: torch.nn.Module, optimizer: Type[optim.Optimizer], save_path,
                 config, input_channel_num_changed=False, final_changed=False, device='cpu'):
        if device == 'cpu':
            self.model = model
        else:
            self.model = nn.DataParallel(model).to(device)
        self.device = device
        self.optimizer = optimizer(self.model.parameters(), lr=config.lr)
        self.start_epoch = 0
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.input_channel_num_changed = input_channel_num_changed
        self.final_changed = final_changed
        self.iter_size = config.iter_size

        self.lr_scheduler = None
        self.lr = config.lr
        self.config = config
        self.optimizer_type = optimizer

    def resume(self, checkpoint_name):
        try:
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint_name), map_location=torch.device('cpu'))
        except:
            print("WARNING: failed to resume from checkpoint: " + os.path.join(self.save_path, checkpoint_name))
            return False

        self.start_epoch = checkpoint['epoch']

        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if self.input_channel_num_changed or self.final_changed:
            skip_layers = self.model.first_layer_params_names if self.input_channel_num_changed else self.model.last_layer_params_names
            print('skipping: ', [k for k in pretrained_dict.keys() if any(s in k for s in skip_layers)])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not any(s in k for s in skip_layers)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

        print("resumed from checkpoint {} on epoch: {}".format(os.path.join(self.save_path, checkpoint_name), self.start_epoch))
        return True

    def calculate_loss_multichannel(self, output, target, meter, training, iter_size):
        bce = F.binary_cross_entropy_with_logits(output, target)
        output = F.sigmoid(output)
        dice = dice_loss(output, target)
        dice_ch2 = dice_loss(output[:,-1,...], target[:,-1,...])
        dice_r = dice_round(output[:,-1,...], target[:,-1,...])

        loss = (self.config.loss['bce'] * bce + self.config.loss['dice'] * (1 - dice)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['dice'] += dice_ch2.data.cpu().numpy()[0] / iter_size
        meter['bce'] += bce.data.cpu().numpy()[0] / iter_size
        meter['dr'] += dice_r.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_ssim_l1_loss(self, output, target, meter, training, iter_size):
        if target.size(1) == 1 and output.size(1) != 1:
            target = target.repeat(1, output.size(1), 1, 1)
        loss = (1 - ssim(output, target)) * self.config.loss['alpha'] + nn.L1Loss()(output, target) * (1- self.config.loss['alpha'])
        if training:
            loss.backward()
        meter['loss'] += loss.data.cpu().numpy()
        return meter

    def calculate_mse_loss(self, output, target, meter, training, iter_size):
        if target.size(1) == 1 and output.size(1) != 1:
            target = target.repeat(1, output.size(1), 1, 1)
        loss = nn.MSELoss()(output, target)
        if training:
            loss.backward()
        meter['loss'] += loss.data.cpu().numpy()
        return meter

    def calculate_loss_single_channel(self, output, target, meter, training, iter_size):
        bce = nn.BCEWithLogitsLoss()(output, target)
        output = F.sigmoid(output)
        dice = dice_loss(output, target)
        dice_r = dice_round(output, target)

        loss = (self.config.loss['bce'] * bce + self.config.loss['dice'] * (1 - dice)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['dice'] += dice.data.cpu().numpy()[0] / iter_size
        meter['bce'] += bce.data.cpu().numpy()[0] / iter_size
        meter['dr'] += dice_r.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_loss_softmax(self, output, target, meter, training, iter_size):
        ce = F.cross_entropy(output, target)
        output = F.softmax(output, dim=1)
        dice_body = dice_loss(output[:,2,...], (target==2).float())
        dice_border = dice_loss(output[:,1,...], (target==1).float())
        dice_r_body = dice_round(output[:,2,...], (target==2).float())
        dice_r_border = dice_round(output[:,1,...], (target==1).float())

        loss = (self.config.loss['ce'] * ce + self.config.loss['dice_body'] * (1 - dice_body) + self.config.loss['dice_border'] * (1 - dice_border)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()
        meter['d_n'] += dice_body.data.cpu().numpy() / iter_size
        meter['d_b'] += dice_border.data.cpu().numpy() / iter_size
        meter['ce'] += ce.data.cpu().numpy() / iter_size
        meter['dr_n'] += dice_r_body.data.cpu().numpy() / iter_size
        meter['dr_b'] += dice_r_border.data.cpu().numpy() / iter_size
        return meter

    def calculate_loss_sigmoid(self, output, target, meter, training, iter_size):
        ce_body = F.binary_cross_entropy_with_logits(output[:,0,...], target[:,2,...])
        ce_border = F.binary_cross_entropy_with_logits(output[:,1,...], target[:,1,...])
        ce = ce_body + ce_border
        output = F.sigmoid(output)
        dice_body = dice_loss(output[:,0,...], target[:,2,...])
        dice_border = dice_loss(output[:,1,...], target[:,1,...])
        dice_r_body = dice_round(output[:,0,...], target[:,2,...])
        dice_r_border = dice_round(output[:,1,...], target[:,1,...])

        loss = (self.config.loss['ce'] * ce + self.config.loss['dice_body'] * (1 - dice_body) + self.config.loss['dice_border'] * (1 - dice_border)) / iter_size

        if training:
            loss.backward()

        meter['loss'] += loss.data.cpu().numpy()[0]
        meter['d_n'] += dice_body.data.cpu().numpy()[0] / iter_size
        meter['d_b'] += dice_border.data.cpu().numpy()[0] / iter_size
        meter['ce'] += ce.data.cpu().numpy()[0] / iter_size
        meter['dr_n'] += dice_r_body.data.cpu().numpy()[0] / iter_size
        meter['dr_b'] += dice_r_border.data.cpu().numpy()[0] / iter_size
        return meter

    def calculate_loss_3ch(self, output, target, meter, training, iter_size):
        ce_body = (F.binary_cross_entropy_with_logits(output[:,2,...], target[:,2,...]) +
                   F.binary_cross_entropy_with_logits(output[:,0,...], target[:,0,...])) / 2
        ce_border = F.binary_cross_entropy_with_logits(output[:,1,...], target[:,1,...])
        ce = ce_body + ce_border
        output = F.sigmoid(output)
        dice_body = (dice_loss(output[:,0,...], target[:,0,...]) + dice_loss(output[:,2,...], target[:,2,...])) / 2
        dice_border = dice_loss(output[:,1,...], target[:,1,...])
        dice_r_body = (dice_round(output[:,0,...], target[:,0,...]) + dice_round(output[:,2,...], target[:,2,...])) / 2
        dice_r_border = dice_round(output[:,1,...], target[:,1,...])

        loss = (self.config.loss['ce'] * ce + self.config.loss['dice_body'] * (1 - dice_body) + self.config.loss['dice_border'] * (1 - dice_border)) / iter_size

        if training:
            loss.backward()

        # meter['loss'] += loss.data.cpu().numpy()[0]
        # meter['d_n'] += dice_body.data.cpu().numpy()[0] / iter_size
        # meter['d_b'] += dice_border.data.cpu().numpy()[0] / iter_size
        # meter['ce'] += ce.data.cpu().numpy()[0] / iter_size
        # meter['dr_n'] += dice_r_body.data.cpu().numpy()[0] / iter_size
        # meter['dr_b'] += dice_r_border.data.cpu().numpy()[0] / iter_size

        meter['loss'] += loss.data.cpu().numpy()
        meter['d_n'] += dice_body.data.cpu().numpy() / iter_size
        meter['d_b'] += dice_border.data.cpu().numpy() / iter_size
        meter['ce'] += ce.data.cpu().numpy() / iter_size
        meter['dr_n'] += dice_r_body.data.cpu().numpy() / iter_size
        meter['dr_b'] += dice_r_border.data.cpu().numpy() / iter_size
        return meter


    def make_step_itersize(self, images, ytrues, training):
        iter_size = self.iter_size
        if training:
            self.optimizer.zero_grad()

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)

        meter = defaultdict(float)
        outputs = []
        for input, target in zip(inputs, targets):
            input = torch.autograd.Variable(input.to(self.device))
            target = torch.autograd.Variable(target.to(self.device))

            if not training:
                with torch.no_grad():
                    output = self.model(input)
                    if self.config.loss["type"] == 'mse':
                        meter = self.calculate_mse_loss(output, target, meter, training, iter_size)
                    elif self.config.loss["type"] == 'ssim+l1':
                        meter = self.calculate_ssim_l1_loss(output, target, meter, training, iter_size)
                    elif self.config.loss["type"] == 'bce':
                        if self.config.activation == 'sigmoid':
                            meter = self.calculate_loss_3ch(output, target, meter, training, iter_size)
                        else:
                            meter = self.calculate_loss_softmax(output, target, meter, training, iter_size)
                    else:
                        raise NotImplementedError
            else:
                output = self.model(input)
                if self.config.loss["type"] == 'mse':
                        meter = self.calculate_mse_loss(output, target, meter, training, iter_size)
                elif self.config.loss["type"] == 'ssim+l1':
                        meter = self.calculate_ssim_l1_loss(output, target, meter, training, iter_size)
                elif self.config.loss["type"] == 'bce':
                    if self.config.activation == 'sigmoid':
                        meter = self.calculate_loss_3ch(output, target, meter, training, iter_size)
                    else:
                        meter = self.calculate_loss_softmax(output, target, meter, training, iter_size)
            
            # meter = self.calculate_loss_single_channel(output, target, meter, training, iter_size)
            #additional metrics
            # for name, func in metrics:
            #     acc = func(output.contiguous(), target.contiguous())
            #     meter[name] += acc.data.cpu().numpy()[0] / iter_size
            outputs.append(output.data)

        if training:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
        return meter, torch.cat(outputs, dim=0)

class PytorchTrain:
    def __init__(self, estimator: Estimator, checkpoint=None, callbacks=None, hard_negative_miner=None):
        self.estimator = estimator

        self.hard_negative_miner = hard_negative_miner
        self.metrics_collection = MetricsCollection()
        self.current_results = {}
        self.training = True

        if checkpoint:
            self.estimator.resume(checkpoint)
        # if self.estimator.model_changed:
        #     callbacks.append(ColdStart(self.estimator.lr, 5, 30, 0.1))

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)

    def _run_one_epoch(self, epoch, loader, training=True):
        avg_meter = defaultdict(float)

        pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc="Epoch {}{}".format(epoch, ' eval' if not training else ""), ncols=0)
        for i, data in pbar:
            self.callbacks.on_batch_begin(i, training)
            meter, ypreds = self._make_step(data, training)
            for k, val in meter.items():
                avg_meter[k] += val
            if training:
                if self.hard_negative_miner is not None:
                    self.hard_negative_miner.update_cache(meter, data)
                    if self.hard_negative_miner.need_iter():
                        self._make_step(self.hard_negative_miner.cache, training)
                        self.hard_negative_miner.invalidate_cache()

            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})
            self.callbacks.on_batch_end(i)
        ret = {k: v / len(loader) for k, v in avg_meter.items()}
        return ret

    def _make_step(self, data, training):
        images = data['image']
        ytrues = data['mask']

        meter, ypreds = self.estimator.make_step_itersize(images, ytrues, training)
        self.current_results['inputs'] = images
        self.current_results['outputs'] = ypreds
        self.current_results['targets'] = ytrues
        self.current_results['image_name'] = data.get('image_name')
        return meter, ypreds

    def fit(self, train_loader, val_loader, nb_epoch):
        self.callbacks.on_train_begin()

        for epoch in range(self.estimator.start_epoch, nb_epoch):
            self.callbacks.on_epoch_begin(epoch)
            self.estimator.model.train()
            self.metrics_collection.train_metrics = self._run_one_epoch(epoch, train_loader, training=True)
            if self.estimator.lr_scheduler is not None and epoch >= self.estimator.config.warmup:
                self.estimator.lr_scheduler.step(epoch)

            with torch.no_grad():
                self.estimator.model.eval()
                self.metrics_collection.val_metrics = self._run_one_epoch(epoch, val_loader, training=False)

            self.callbacks.on_epoch_end(epoch)

            if self.metrics_collection.stop_training:
                break

        self.callbacks.on_train_end()
        
        
class MetricsCollection:
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}

def predict8tta(model, batch, activation, scale_tta=False):
    ret = []
    d = next(model.parameters()).device
    batch = batch.to(d)
    for cls in TTA:
        with torch.no_grad():
            ret.append(cls(activation)(model, batch))

    if scale_tta:
        for scale in [0.8, 1.25]:
            data = np.moveaxis(np.squeeze(batch.numpy()[0]), 0, -1)
            srows, scols = data.shape[:2]
            data = cv2.resize(data, (0, 0), fx=scale, fy=scale)
            rows, cols = data.shape[:2]
            data = cv2.copyMakeBorder(data, 0, (32-rows%32), 0, (32-cols%32), cv2.BORDER_REFLECT)
            data = np.expand_dims(np.moveaxis(data, -1, 0), 0)
            data = torch.from_numpy(data)
            for cls in TTA:
                with torch.no_grad():
                    r = (cls(activation)(model, data))
                    r = np.moveaxis(np.squeeze(r), 0, -1)
                    r = r[:rows, :cols, ...]
                    r = cv2.resize(r, (scols, srows))
                    r = np.expand_dims(np.moveaxis(r, -1, 0), 0)
                    ret.append(r)
    return np.moveaxis(np.mean(ret, axis=0), 1, -1)
