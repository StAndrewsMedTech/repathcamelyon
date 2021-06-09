from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from sklearn.utils.validation import check_is_fitted

import repath.data.datasets.bloodmucus as bloodmucus
from repath.data.slides.isyntax import Slide
from repath.preprocess.tissue_detection.multiscale_features import multiscale_basic_features, multiscale_basic_features_all
from repath.utils.metrics import conf_mat_plot_heatmap



def get_slides_annots(dset, level, default_label="tissue"):
    thumbz = []
    annotz = []

    for dd in dset:
        slide_path, annot_path, _, _ = dd
        with Slide(dset.root / slide_path) as slide:
            thumb = slide.get_thumbnail(level)
            thumbz.append(thumb)
            annotations = dset.load_annotations(annot_path, default_label)
            annot = annotations.render(thumb.shape[0:2], 2**level)
            annotz.append(annot)

    return thumbz, annotz





def apply_tissue_detection(thumbz, tissue_detector):

    filtered_thumbz = []

    for tt in thumbz:
        tissue_mask = tissue_detector(tt)
        three_d_mask = np.expand_dims(tissue_mask, axis=-1)
        three_d_mask = np.dstack((three_d_mask, three_d_mask, three_d_mask))
        filtered_thumb = np.where(np.logical_not(three_d_mask), 255, tt)
        filtered_thumbz.append(filtered_thumb)

    return filtered_thumbz


def get_features(filtered_thumb, sigma_min = 1, sigma_max=16, edges=False):
    features_func = partial(multiscale_basic_features,
                            intensity=True, edges=edges, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            multichannel=True)
    features = features_func(filtered_thumb)
    return features


def get_features_all(filtered_thumb, sigma_min = 1, sigma_max=16, edges=False):
    features_func = partial(multiscale_basic_features_all,
                            intensity=True, edges=edges, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            multichannel=True)
    features = features_func(filtered_thumb)
    return features


def get_features_list(filtered_thumbz, sigma_min = 1, sigma_max=16, edges=False):

    featz = []
    for idx, tt in enumerate(filtered_thumbz):
        print(idx)
        features = get_features(tt, sigma_min, sigma_max, edges)
        features = np.array(features, dtype=np.float32)
        featz.append(features)

    return featz


def get_features_list_all(filtered_thumbz, sigma_min = 1, sigma_max=16, edges=False):

    featz = []
    for idx, tt in enumerate(filtered_thumbz):
        print(idx)
        features = get_features_all(tt, sigma_min, sigma_max, edges)
        features = np.array(features, dtype=np.float32)
        featz.append(features)

    return featz


def fit_segmenter_multi(labels_list, features_list, clf):
    """Segmentation using labeled parts of the image and a classifier.
    Parameters
    ----------
    labels : ndarray of ints
        Image of labels. Labels >= 1 correspond to the training set and
        label 0 to unlabeled pixels to be segmented.
    features : ndarray
        Array of features, with the first dimension corresponding to the number
        of features, and the other dimensions correspond to ``labels.shape``.
    clf : classifier object
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    Returns
    -------
    clf : classifier object
        classifier trained on ``labels``
    Raises
    ------
    NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
    """
    training_data_all = np.concatenate(features_list, axis=0)
    training_labels_all = np.concatenate(labels_list, axis=0)
    #for idx, labels in enumerate(labels_list):
        # mask = labels > 0
        # training_data = features_list[idx][mask]
        #training_data = features_list[idx]
        #if idx == 0:
        #    training_data_all = training_data
        #else:
        #    training_data_all = np.vstack((training_data_all, training_data))
        # training_labels = labels[mask].ravel()
        #training_labels_all = np.hstack((training_labels_all, labels))
    print(training_data_all.dtype, training_labels_all.dtype)
    clf.fit(training_data_all, training_labels_all)
    print(clf)
    return clf


def predict_segmenter(features, clf):
    """Segmentation of images using a pretrained classifier.
    Parameters
    ----------
    features : ndarray
        Array of features, with the last dimension corresponding to the number
        of features, and the other dimensions are compatible with the shape of
        the image to segment, or a flattened image.
    clf : classifier object
        trained classifier object, exposing a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
        classifier must be already trained, for example with
        :func:`skimage.segmentation.fit_segmenter`.
    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier.
    """
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))

    try:
        predicted_labels = clf.predict(features)
    except NotFittedError:
        raise NotFittedError(
            "You must train the classifier `clf` first"
            "for example with the `fit_segmenter` function."
        )
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(
                err.args[0] + '\n' +
                "Maybe you did not use the same type of features for training the classifier."
                )
    output = predicted_labels.reshape(sh[:-1])
    return output


def blood_mucus_patching(patch):
    if np.sum(patch) == 0:
        lab_out = 0
    elif np.sum(np.equal(patch, 1)) > 0:
        lab_out = 1
    else:
        lab_out = 2

    return lab_out


def pool_blood_mucus(A, kernel_size, stride, padding):
    """
    2D Pooling

    Taken from https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )

    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )

    A_w = A_w.reshape(-1, *kernel_size)

    A_flat = np.reshape(A_w, (A_w.shape[0], kernel_size[0]**2))
    pooled_vals = np.apply_along_axis(blood_mucus_patching, 1, A_flat)
    output_new = pooled_vals.reshape(output_shape)

    return output_new
    

def calc_confusion_mat_3class(true_img, pred_img):
    tru0_pred0 = np.sum(np.logical_and(true_img == 0, pred_img == 0))
    tru0_pred1 = np.sum(np.logical_and(true_img == 0, pred_img == 1))
    tru0_pred2 = np.sum(np.logical_and(true_img == 0, pred_img == 2))
    tru1_pred0 = np.sum(np.logical_and(true_img == 1, pred_img == 0))
    tru1_pred1 = np.sum(np.logical_and(true_img == 1, pred_img == 1))
    tru1_pred2 = np.sum(np.logical_and(true_img == 1, pred_img == 2))
    tru2_pred0 = np.sum(np.logical_and(true_img == 2, pred_img == 0))
    tru2_pred1 = np.sum(np.logical_and(true_img == 2, pred_img == 1))
    tru2_pred2 = np.sum(np.logical_and(true_img == 2, pred_img == 2))

    cm = [tru0_pred0, tru0_pred1, tru0_pred2, tru1_pred0, tru1_pred1, tru1_pred2, tru2_pred0, tru2_pred1, tru2_pred2]
    return cm


def calc_confusion_mat_2class(true_img, pred_img):
    tru0_pred0 = np.sum(np.logical_and(true_img == 0, np.logical_not(pred_img == 1)))
    tru0_pred1 = np.sum(np.logical_and(true_img == 0, pred_img == 1))
    tru1_pred0 = np.sum(np.logical_and(true_img == 1, np.logical_not(pred_img == 1)))
    tru1_pred1 = np.sum(np.logical_and(true_img == 1, pred_img == 1))
    cm2 = [tru0_pred0, tru0_pred1, tru1_pred0, tru1_pred1]
    return cm2

def save_confusion_mat(cm, save_dir, exp_name):
    cm_pd = np.array(cm, dtype=np.uint)
    cm_pd = np.expand_dims(cm_pd, axis=0)
    title_cm = 'confusion matrix blood mucus ' + exp_name
    if len(cm) == 4:
        cm_pd = pd.DataFrame(cm_pd, columns=['tn', 'fp', 'fn', 'tp'])
        cm_pd.to_csv(save_dir / 'confusion_matrix_2class.csv')
        display_labels = ['not tissue', 'tissue']
        cm_np = np.reshape(cm, (2,2))
        cm_np = np.array(cm_np,dtype=np.uint)
        cm_out = conf_mat_plot_heatmap(cm_np, display_labels, title_cm)
        cm_out.get_figure().savefig(save_dir / 'confusion_matrix_2class.png')
    else:
        cm_pd = pd.DataFrame(cm_pd, columns=['true0pred0', 'true0pred1', 'true0pred2', 'true1pred0', 'true1pred1', 'true1pred2', 'true2pred0', 'true2pred1', 'true2pred2'])
        cm_pd.to_csv(save_dir / 'confusion_matrix_3class.csv')
        display_labels = ['background', 'tissue', 'blood or mucus']
        cm_np = np.reshape(cm, (3,3))
        cm_np = np.array(cm_np,dtype=np.uint)
        cm_out = conf_mat_plot_heatmap(cm_np, display_labels, title_cm)
        cm_out.get_figure().savefig(save_dir / 'confusion_matrix_3class.png')


def flatten_mask_annots(annots, image):
    rr = image[:, :, 0] == 255
    gg = image[:, :, 0] == 255
    bb = image[:, :, 0] == 255
    mask = np.logical_not(np.logical_and(np.logical_and(rr, gg), bb))
    annots = annots[mask]
    return annots

def flatten_mask_annots_list(annotz, thumbz):
    annots_out = []
    for img, ann in zip(thumbz, annotz):
        annot = flatten_mask_annots(ann, img)
        annots_out.append(annot)
    return annots_out