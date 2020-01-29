import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

eps = 1e-3

def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice_loss(preds, trues, is_average=is_average)

def multi_class_dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return multi_class_dice(preds, trues, is_average=is_average)

def dice_loss(preds, trues, weight=None, is_average=True):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    if is_average:
        score = scores.sum()/num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


def per_class_dice(preds, trues, weight=None, is_average=True):
    loss = []
    for idx in range(1, preds.shape[1]):
        loss.append(dice_loss(preds[:,idx,...].contiguous(), (trues==idx).float().contiguous(), weight, is_average))
    return loss


def multi_class_dice(preds, trues, weight=None, is_average=True):
    channels = per_class_dice(preds, trues, weight, is_average)
    return sum(channels) / len(channels)


def jaccard_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return jaccard(preds, trues, is_average=is_average)


def jaccard(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return dice_loss(input, target, self.weight, self.size_average)

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        return jaccard(input, target, self.weight, self.size_average)

class BCEDiceJaccardLoss(nn.Module):
    def __init__(self, weights, weight=None, size_average=True):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.jacc = JaccardLoss()
        self.dice = DiceLoss()
        self.mapping = {'bce': self.bce,
                        'jacc': self.jacc,
                        'dice': self.dice}
        self.values = {}

    def forward(self, input, target):
        loss = 0
        sigmoid_input = torch.sigmoid(input)
        for k, v in self.weights.items():
            if not v: 
                continue
            val = self.mapping[k](input if k == 'bce' else sigmoid_input, target)
            self.values[k] = val
            if k != 'bce':
                loss += self.weights[k] * (1 - val)
            else:
                loss += self.weights[k] * val
        return loss


# from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

