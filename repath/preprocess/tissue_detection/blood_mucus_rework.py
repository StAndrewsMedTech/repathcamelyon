from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from repath.data.datasets import Dataset
from repath.preprocess.tissue_detection import TissueDetector
from repath.preprocess.tissue_detection.pixel_feature_detector import PixelFeatureDetector

def get_slides_annots(dset: Dataset, level: int, default_label="tissue") -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Creates a list of thumbnails of slides and annotations in a dataset.

    It is used to load thumbnails of a set of slides and their annotations.

    Args:
        dset (Dataset): the dataset containing slides and annotation paths
        level (int):  the level to create thumbnails at
        default_label (string, optional): label to use for areas in annotation thumbnail for which no annotation is provided

    Returns:
        A tuple of two lists 
        the first containing thumbnails of all slides in dataset at level
        the second containing thumbnails of annotations in dataset at level
    """
    thumbz = []
    annotz = []

    for dd in dset:
        slide_path, annot_path, _, _ = dd
        with dset.slide_cls(dset.root / slide_path) as slide:
            thumb = slide.get_thumbnail(level)
            thumbz.append(thumb)
            annotations = dset.load_annotations(annot_path, default_label)
            annot = annotations.render(thumb.shape[0:2], 2**level)
            annotz.append(annot)

    return thumbz, annotz


def set_background_to_white(thumbz: List[np.ndarray], tissue_detector:TissueDetector) -> List[np.ndarray]:
    """ Converts background determined by a tissue detector to white.

    For a given list of thumbnails and a tissue detector converts background areas of the thumbnails to pure white

    Args:
        thumbz (List): A list of ndarrays each of white is a thumbnail image of a slide
        tissue_detector (TissueDetector): a tissue detector to determine background

    Returns:
        A list of ndarray arrays of slide thumbnails where background areas have been set to pure white
    """

    filtered_thumbz = []

    for tt in thumbz:
        tissue_mask = tissue_detector(tt)
        three_d_mask = np.expand_dims(tissue_mask, axis=-1)
        three_d_mask = np.dstack((three_d_mask, three_d_mask, three_d_mask))
        filtered_thumb = np.where(np.logical_not(three_d_mask), 255, tt)
        filtered_thumbz.append(filtered_thumb)

    return filtered_thumbz


def calculate_annotation_class_sizes(annotz: List[np.ndarray]) -> pd.DataFrame:
    """ calculates a dataframe with pixels per class

    For each annotation thumbnail in a list calculates how many pixels there 
    are of each class and saves it in a dataframe

    Args:
        annotz (List): A list of ndarrays each of which is a thumbnail of an annotation

    Returns:
        A pandas dataframe where each row contains the number of pixels of each class 
    """

    class_table = pd.DataFrame(index=range(len(annotz)),columns=["tissue", "blood", "mucus", "blood_mucus"])

    for idx, ann in enumerate(annotz):
        class_table.loc[idx, 'tissue'] = np.sum(ann == 1)
        class_table.loc[idx, 'blood'] = np.sum(ann == 2)
        class_table.loc[idx, 'mucus'] = np.sum(ann == 3)
        class_table.loc[idx, 'blood_mucus'] = np.sum(ann == 4)

    return class_table


def calculate_annotation_class_sample_size(class_table: pd.DataFrame, classsamp: int = 100000) -> List[int]:
    """ calculates per slide sample size

    For a given sample size of each class calculates how many slides contain that class 
    and therefore the number of samples per slide that should be taken. 

    Args:
        class_table (pd.Dataframe): A dataframe with one row per slide 
            with the the number of annotated pixels per class
        classsamp (int): the total number of samples required per class

    Returns:
        A list of four integers one for each class of how many samples to take per slide
        order in list is tissue, blood, mucus, blood_mucus
    """
    
    ntiss = int(classsamp / np.sum(class_table['tissue'] > 0))
    nblod = int(classsamp / np.sum(class_table['blood'] > 0))
    nmucs = int(classsamp / np.sum(class_table['mucus'] > 0))
    nblmc = int(classsamp / np.sum(class_table['blood_mucus'] > 0))
    nclass = [ntiss, nblod, nmucs, nblmc]

    return nclass


def sample_features_from_slides(slide_thumbz: List[np.ndarray], slide_annotz: List[np.ndarray], nclass: List[int], pixel_feat_detector: PixelFeatureDetector) -> Tuple[np.ndarray, np.ndarray]:
    """ calculates features for a list of thumbnails of slides

    For each slide in the list of slide thumbnails:
    Calculates a set of features for each pixel using the input pixel feature detector.
    Get the matching annotations for each pixel.
    Samples n of each class of annotation as defined using nclass of the features. 
    Outputs for each slide an array of features for all classes and matching annotations

    Args:
        slide_thumbz (List of ndarrays): A list of ndarrays one for each slide thumbnail 
        slide_annotz (List of ndarrays): A list of ndarrays containing thumbnails of annotations for each slide
        nclass (list of int): the total number of samples required per class
        pixel_feat_detector (PixelFeatureDetector): the pixel feature detector class defines 
            how features are to be extracted frome ach slide

    Returns:
        A numpy array of features sampled from slides and a matching numpy array of classes to predict
    """
    for idx, sl in enumerate(slide_thumbz):
        print(idx + 1, "of", len(slide_thumbz))
        feat = pixel_feat_detector(sl)
        feat_reshape = feat.reshape(-1, feat.shape[-1])
        annz = slide_annotz[idx].flatten()
        for cl in range(len(nclass)):
            class_mask = annz == (cl + 1)
            class_feat = feat_reshape[class_mask]
            class_samp = class_feat[np.random.choice(class_feat.shape[0], min(nclass[cl], class_feat.shape[0]), replace=False), :]
            class_annot = np.repeat((cl + 1), class_samp.shape[0])
            class_annot = np.reshape(class_annot, (class_annot.shape[0], 1))
            if cl == 0:
                slide_feat = class_samp
                slide_annot = class_annot
            else:
                slide_feat = np.vstack((slide_feat, class_samp))
                slide_annot = np.vstack((slide_annot, class_annot))
        if idx == 0:
            all_feat = slide_feat
            all_annot = slide_annot
        else:
            all_feat = np.vstack((all_feat, slide_feat))
            all_annot = np.vstack((all_annot, slide_annot))

    return all_feat, all_annot


