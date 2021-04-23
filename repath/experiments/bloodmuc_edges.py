import json
from joblib import dump, load
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

import repath.data.datasets.bloodmucus as bloodm
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.tissue_detection import TissueDetectorGreyScale, SizedClosingTransform, FillHolesTransform
from repath.preprocess.tissue_detection.blood_mucus import get_slides_annots, apply_tissue_detection, get_features_list, fit_segmenter_multi, predict_segmenter, get_features, pool_blood_mucus, get_annot_areas, calc_confusion_mat_2class, calc_confusion_mat_3class, save_confusion_mat
from repath.utils.convert import get_concat_h, get_concat_v
from repath.utils.paths import project_root
from repath.utils.seeds import set_seed

from sklearn.utils.validation import check_is_fitted

"""
Global stuff
"""
experiment_name = "bloodmucus_edges"
experiment_root = project_root() / "experiments" / experiment_name

global_seed = 123

### temp for debugging
# split between train and predict


def train_classifier() -> None:
    ### generic can be reused for multiple classifiers
    # set values
    set_seed(global_seed)
    level_label = 6

    # read in slides and annotations for training
    dset = bloodm.training()
    thumbz, annotz = get_slides_annots(dset, level_label)

    # apply tissue detection
    morphology_transform1 = SizedClosingTransform(level_in=level_label)
    morphology_transform2 = FillHolesTransform(level_in=level_label)
    morphology_transforms = [morphology_transform1, morphology_transform2]
    tissue_detector = TissueDetectorGreyScale(grey_level=0.85, morph_transform = morphology_transforms)
    filtered_thumbz = apply_tissue_detection(thumbz, tissue_detector)

    ### classifier specific 
    # get features
    featz = get_features_list(filtered_thumbz, edges=True)

    # apply classifier
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
    clf = fit_segmenter_multi(annotz, featz, clf)
    print("save: ", experiment_root, check_is_fitted(clf))
    experiment_root.mkdir(parents=True, exist_ok=True)
    dump(clf, experiment_root / 'rforest.joblib') 


def predict_images() -> None:
    ### generic can be reused for multiple classifiers
    # set values
    set_seed(global_seed)
    level_label = 6
    thumb_level = 7

    # read in slides and annotations for training
    dset = bloodm.validation()
    # assume unlabeled annotations are the annotated area label as background for now
    thumbz, annotz = get_slides_annots(dset, level_label, "background")
    thumbz_out, annotz_out = get_slides_annots(dset, thumb_level, "background")
    # get unlabled annotations as annotated areas
    annot_areaz = get_annot_areas(dset, thumb_level)


    # apply tissue detection
    morphology_transform1 = SizedClosingTransform(level_in=level_label)
    morphology_transform2 = FillHolesTransform(level_in=level_label)
    morphology_transforms = [morphology_transform1, morphology_transform2]
    tissue_detector = TissueDetectorGreyScale(grey_level=0.85, morph_transform = morphology_transforms)
    filtered_thumbz = apply_tissue_detection(thumbz, tissue_detector)

    # get features
    featz = get_features_list(filtered_thumbz, edges=True)

    # set output directories
    labels_dir = experiment_root / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    tissue_dir = experiment_root / 'tissue'
    tissue_dir.mkdir(parents=True, exist_ok=True)
    bloodm_dir = experiment_root / 'bloodm'
    bloodm_dir.mkdir(parents=True, exist_ok=True)
    annot_out_dir = experiment_root / 'annot_out'
    annot_out_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir = experiment_root / 'mosaic'
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    ### classifier specific 
    # load classifier
    clf = load(experiment_root / 'rforest.joblib')
    print(clf, check_is_fitted(clf))
    # settings for classifier
    patch_level = 0
    patch_size = 2**thumb_level
    stride = patch_size
    scale_factor = 2 ** (level_label - patch_level)
    kernel_size = int(patch_size / scale_factor)
    label_level_stride = int(stride / scale_factor)
    confusion_matrix_2class = np.zeros((len(thumbz), 4))
    confusion_matrix_3class = np.zeros((len(thumbz), 9))

    for idx, thumb in enumerate(thumbz):
        ### classifier specific
        features = get_features(thumb, edges=True)
        output = predict_segmenter(features, clf)
        tissue_mask = tissue_detector(thumb)
        filtered_output = np.where(np.logical_not(tissue_mask), 0, output)
        output_labels = pool_blood_mucus(filtered_output, kernel_size, label_level_stride, 0)
        
        ### generic can be reused for multiple classifiers
        # create image with just tissue 
        thumb_out = thumbz_out[idx]
        if thumb_out.shape[0:2] != output_labels.shape:
            output_labels = output_labels[0:thumb_out.shape[0], 0:thumb_out.shape[1]]
        tissue_output = np.where(np.expand_dims(output_labels, axis=-1) == 1, thumb_out, 255)
        # create image with jsut blood mucus
        bloodm_output = np.where(np.expand_dims(output_labels, axis=-1) == 2, thumb_out, 255)
        # convert to pil images and write
        filename = str(idx) + '.png'
        labels_image = Image.fromarray(np.array(output_labels*100, dtype=np.uint8))
        labels_image.save(labels_dir / filename)
        # make annot image coloured
        annot_out = annotz_out[idx]
        annot_out = np.expand_dims(annot_out, axis=-1)
        annot_out_cl = np.dstack((annot_out, annot_out, annot_out))
        tissue = np.array([255,255,255]).reshape((1,1,3))
        blood = np.array([255,0,0]).reshape((1,1,3))
        mucus = np.array([255,208,182]).reshape((1,1,3))
        blmuc = np.array([186,85,211]).reshape((1,1,3))
        annot_out_cl = np.where(annot_out == 1, tissue, annot_out_cl)
        annot_out_cl = np.where(annot_out == 2, blood, annot_out_cl)
        annot_out_cl = np.where(annot_out == 3, mucus, annot_out_cl)
        annot_out_cl = np.where(annot_out == 4, blmuc, annot_out_cl)

        annot_out_image = Image.fromarray(np.array(annot_out_cl, dtype=np.uint8))
        annot_out_image.save(annot_out_dir / filename)
        tissue_image = Image.fromarray(np.array(tissue_output, dtype=np.uint8))
        tissue_image.save(tissue_dir / filename)
        bloodm_image = Image.fromarray(np.array(bloodm_output, dtype=np.uint8))
        bloodm_image.save(bloodm_dir / filename)
        top_row = get_concat_h(labels_image, tissue_image)
        low_row = get_concat_h(bloodm_image, annot_out_image)
        get_concat_v(top_row, low_row).save(mosaic_dir / filename)
        print(idx)

        # cut down to annotated areas only
        annot_area = annot_areaz[idx]
        pixvals = np.where(annot_area > 0)
        minrw = min(pixvals[0])
        maxrw = max(pixvals[0])+1
        mincl = min(pixvals[1])
        maxcl = max(pixvals[1])+1
        cd_annots = annot_out[minrw:maxrw, mincl:maxcl, 0]
        cd_annots = np.where(cd_annots > 1, 2, cd_annots)
        cd_preds = output_labels[minrw:maxrw, mincl:maxcl]
        cm3cl = calc_confusion_mat_3class(cd_annots, cd_preds)
        cm2cl = calc_confusion_mat_2class(cd_annots, cd_preds)
        confusion_matrix_3class[idx, :] = cm3cl
        confusion_matrix_2class[idx, :] = cm2cl

    cm_3class_all = np.sum(confusion_matrix_3class, axis=0)
    save_confusion_mat(cm_3class_all, experiment_root, experiment_name)
    cm_2class_all = np.sum(confusion_matrix_2class, axis=0)
    save_confusion_mat(cm_2class_all, experiment_root, experiment_name)
