import numpy as np
import pandas as pd
from scipy.misc import imread
import cv2
import os
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from skimage import measure
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(lab_img):
    # lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def my_watershed(what, mask1, mask2):
    # markers = ndi.label(mask2, output=np.uint32)[0]
    # big_seeds = watershed(what, markers, mask=mask1, watershed_line=False)
    # m2 = mask1 - (big_seeds > 0)
    # mask2 = mask2 | m2

    markers = ndi.label(mask2, output=np.uint32)[0]
    labels = watershed(what, markers, mask=mask1, watershed_line=True)
    # labels = watershed(what, markers, mask=mask1, watershed_line=False)
    return labels


def calc_score(labels, y_pred):
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    #    print("Number of true objects:", true_objects)
    #    print("Number of predicted objects:", pred_objects)
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    #    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        #        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    #    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec), prec


def wsh(mask_img, threshold, border_img, seeds):
    img_copy = np.copy(mask_img)
    m = seeds * border_img# * dt
    img_copy[m <= threshold + 0.35] = 0
    img_copy[m > threshold + 0.35] = 1
    img_copy = img_copy.astype(np.bool)
    img_copy = remove_small_objects(img_copy, 10).astype(np.uint8)

    mask_img[mask_img <= threshold] = 0
    mask_img[mask_img > threshold] = 1
    mask_img = mask_img.astype(np.bool)
    mask_img = remove_small_holes(mask_img, 1000)
    mask_img = remove_small_objects(mask_img, 8).astype(np.uint8)
    # cv2.imwrite('t.png', (mask_img * 255).astype(np.uint8))
    # cv2.imwrite('t2.png', (img_copy * 255).astype(np.uint8))
    labeled_array = my_watershed(mask_img, mask_img, img_copy)
    return labeled_array

def postprocess_victor(pred):
    av_pred = pred / 255.
    av_pred = av_pred[..., 2] * (1 - av_pred[..., 1])
    av_pred = 1 * (av_pred > 0.5)
    av_pred = av_pred.astype(np.uint8)

    y_pred = measure.label(av_pred, neighbors=8, background=0)
    props = measure.regionprops(y_pred)
    for i in range(len(props)):
        if props[i].area < 12:
            y_pred[y_pred == i + 1] = 0
    y_pred = measure.label(y_pred, neighbors=8, background=0)

    nucl_msk = (255 - pred[..., 2])
    nucl_msk = nucl_msk.astype('uint8')
    y_pred = watershed(nucl_msk, y_pred, mask=((pred[..., 2] > 80)), watershed_line=True)
    return y_pred


test_dir = '../results_test/dpn_softmax_f0fold0_'
im_names = os.listdir(test_dir)
test_ids = [os.path.splitext(i)[0] for i in im_names]
preds_test = [imread(os.path.join(test_dir, im, 'nuclei_weighted_boarder.png'), mode='RGB') for im in im_names]
for n, id_ in enumerate(test_ids):
    print(os.path.join(test_dir, im_names[n], 'nuclei_weighted_boarder_masks.png'))
    test_img = wsh(preds_test[n][...,2] / 255., 0.3, 1 - preds_test[n][...,1] / 255., preds_test[n][...,2] / 255)
    cv2.imwrite(os.path.join(test_dir, im_names[n], 'nuclei_weighted_boarder_masks.png'), (test_img > 0).astype(np.uint8) * 255)

    test_img2 = postprocess_victor(preds_test[n])
    cv2.imwrite(os.path.join(test_dir, im_names[n], 'nuclei_weighted_boarder_masks_victor.png'), (test_img > 0).astype(np.uint8) * 255)
