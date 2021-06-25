import json
from joblib import dump, load
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

import repath.data.datasets.bloodmucus as bloodm
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.tissue_detection import TissueDetectorGreyScale, SizedClosingTransform, FillHolesTransform
from repath.preprocess.tissue_detection.pixel_feature_detector import EdgeFeature, TextureFeature, PixelFeatureDetector
from repath.preprocess.tissue_detection.blood_mucus_rework import get_slides_annots, set_background_to_white, \
    calculate_annotation_class_sizes, calculate_annotation_class_sample_size, sample_features_from_slides
from repath.postprocess.blood_mucus_results import pool_blood_mucus, get_annot_areas, \
    calc_confusion_mat_2class, calc_confusion_mat_3class, save_confusion_mat
from repath.utils.convert import get_concat_h, get_concat_v
from repath.utils.paths import project_root
from repath.utils.seeds import set_seed


"""
Global stuff
"""
experiment_name = "bloodmucus_sigma0"
experiment_root = project_root() / "experiments" / experiment_name

global_seed = 123
feature_level = 4

# define tissue detector
morphology_transform1 = SizedClosingTransform(level_in=feature_level)
morphology_transform2 = FillHolesTransform(level_in=feature_level)
morphology_transforms = [morphology_transform1, morphology_transform2]
tissue_detector = TissueDetectorGreyScale(grey_level=0.85, morph_transform = morphology_transforms)

# define pixel feature detector
features_list = [EdgeFeature(), TextureFeature()]
pixel_feature_detector = PixelFeatureDetector(features_list=features_list, sigma_min = 1, sigma_max = 16, raw=True)

def train_classifier() -> None:
    # set values
    set_seed(global_seed)
    samples_per_class = 100000

    # read in slides and annotations for training
    dset = bloodm.training()
    thumbz, annotz = get_slides_annots(dset, feature_level, default_label="background")
    
    # apply tissue detection
    filtered_thumbz = set_background_to_white(thumbz, tissue_detector)

    # take samples
    class_table = calculate_annotation_class_sizes(annotz)
    nclass = calculate_annotation_class_sample_size(class_table, samples_per_class)

    # sample features and annotations
    feats_sample, label_sample = sample_features_from_slides(filtered_thumbz, annotz, nclass, pixel_feature_detector)
    print(feats_sample.shape, label_sample.shape)
    # train classifier
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.25)
    clf.fit(feats_sample, label_sample.ravel())
    print("save: ", experiment_root, check_is_fitted(clf))
    experiment_root.mkdir(parents=True, exist_ok=True)
    dump(clf, experiment_root / 'rforest.joblib') 


def predict_images() -> None:
    # set values
    set_seed(global_seed)
    output_level = 7

    # read in validation dataset
    dset = bloodm.validation()

    # read in slides and for extracting features
    thumbz, annotz = get_slides_annots(dset, feature_level, default_label="background")
    filtered_thumbz = set_background_to_white(thumbz, tissue_detector)

    # read in slides and annotations for plotting and calculating results
    thumbz_out, annotz_out = get_slides_annots(dset, output_level, default_label="background")
    filtered_thumbz_out = set_background_to_white(thumbz_out, tissue_detector)
    annot_areaz = get_annot_areas(dset, output_level)

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
    thumbs_dir = experiment_root / 'thumbnails'
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    backgr_dir = experiment_root / 'background'
    backgr_dir.mkdir(parents=True, exist_ok=True)

    # load classifier
    clf = load(experiment_root / 'rforest.joblib')
    print(clf, check_is_fitted(clf))

    # settings for pooling from feature level to output level
    patch_level = 0
    patch_size = 2**output_level
    stride = patch_size
    scale_factor = 2 ** (feature_level - patch_level)
    kernel_size = int(patch_size / scale_factor)
    label_level_stride = int(stride / scale_factor)

    # create blank confusion matrices
    confusion_matrix_2class = np.zeros((len(thumbz), 4))
    confusion_matrix_3class = np.zeros((len(thumbz), 9))

    for idx, thumb in enumerate(filtered_thumbz):
        print(idx, "of", len(filtered_thumbz))
        # get features from pixel feature detector (shape is thumb rows, thumb columns, nfeatures)
        features = pixel_feature_detector(thumb)
        # flatten so shape is 2d (thumb rows * thumb columns, nfeatures)
        features_reshape = features.reshape(-1, features.shape[-1])
        
        # predict from features
        output = clf.predict(features_reshape)
        
        # reshape back to thumb shape
        labels_image = np.reshape(output, (thumb.shape[:-1]))

        # use tissue detector to set background values to zero
        tissue_mask = tissue_detector(thumb)
        filtered_labels_image = np.where(np.logical_not(tissue_mask), 0, labels_image)

        # change to output size by pooling values 
        output_labels = pool_blood_mucus(filtered_labels_image, kernel_size, label_level_stride, 0)
        # output labels, background = 0, tissue = 1, blood or mucus = 2

        # get output thumbnail
        thumb_out = thumbz_out[idx]
        filtered_thumb_out = filtered_thumbz_out[idx]
        # adjust size sometimes pixel size off by one ### HACK
        if thumb_out.shape[0:2] != output_labels.shape:
            output_labels = output_labels[0:thumb_out.shape[0], 0:thumb_out.shape[1]]

        # create output image with just tissue
        tissue_output = np.where(np.expand_dims(output_labels, axis=-1) == 1, filtered_thumb_out, 255)
        # create output image with just blood mucus
        bloodm_output = np.where(np.expand_dims(output_labels, axis=-1) == 2, filtered_thumb_out, 255)

        # colour annotations for output
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

        # output images, convert to PIL and save
        filename = str(idx) + '.png'
        # output thumbnail unprocessed
        thumb_out_image = Image.fromarray(np.array(thumb_out, dtype=np.uint8))
        thumb_out_image.save(thumbs_dir / filename)
        # output thumbnail with background removed
        filtered_thumb_out_image = Image.fromarray(np.array(filtered_thumb_out, dtype=np.uint8))
        filtered_thumb_out_image.save(backgr_dir / filename)
        # output pathologist annotations
        annot_out_image = Image.fromarray(np.array(annot_out_cl, dtype=np.uint8))
        annot_out_image.save(annot_out_dir / filename)
        # output stuff removed by blood and mucus detector
        bloodm_image = Image.fromarray(np.array(bloodm_output, dtype=np.uint8))
        bloodm_image.save(bloodm_dir / filename)
        # output remaining tissue
        tissue_image = Image.fromarray(np.array(tissue_output, dtype=np.uint8))
        tissue_image.save(tissue_dir / filename)
        # output predicted labels
        output_labels_image = Image.fromarray(np.array(output_labels*100, dtype=np.uint8))
        output_labels_image.save(labels_dir / filename)

        # create and output mosaic of outputs
        top_row = get_concat_h(get_concat_h(thumb_out_image, filtered_thumb_out_image), annot_out_image)
        low_row = get_concat_h(get_concat_h(bloodm_image, tissue_image), output_labels_image)
        get_concat_v(top_row, low_row).save(mosaic_dir / filename)

        # calculate numeric results        
        # get area which is fuly annotated and calculate position
        annot_area = annot_areaz[idx]
        pixvals = np.where(annot_area > 0)
        minrw = min(pixvals[0])
        maxrw = max(pixvals[0])+1
        mincl = min(pixvals[1])
        maxcl = max(pixvals[1])+1

        # cutdown thumbnail to just annotated_area
        cd_thumb = thumb_out[minrw:maxrw, mincl:maxcl, :]
        # cutdown thumbnail with background removed to just annotated area
        cd_fil_thumb = filtered_thumb_out[minrw:maxrw, mincl:maxcl, :]
        # cutdown pathologist annotation thumbnail to just annotated area
        cd_annots = annot_out[minrw:maxrw, mincl:maxcl, 0]
        # cut down stuff removed by blood mucus detector to just annotated area
        cd_bloodm = bloodm_output[minrw:maxrw, mincl:maxcl, :]
        # cut down remaining tissue to annotated area
        cd_tissue = tissue_output[minrw:maxrw, mincl:maxcl, :]
        # cutdown predicted labels thumbnail to just annotated area
        cd_preds = output_labels[minrw:maxrw, mincl:maxcl]

        # change labels of pathologist annotation so blood and mucus classes as grouped together
        cd_annots = np.where(cd_annots > 1, 2, cd_annots)

        # calculate confusion matrices for this slide output is 1d vector
        # 3 class background, blood_mucus, tissue
        cm3cl = calc_confusion_mat_3class(cd_annots, cd_preds)
        # 2 class tissue, not tissue
        cm2cl = calc_confusion_mat_2class(cd_annots, cd_preds)

        # add to array of confusion matrices for all slides
        confusion_matrix_3class[idx, :] = cm3cl
        confusion_matrix_2class[idx, :] = cm2cl

        # output cutdown images, convert to PIL and save
        filename_cd = str(idx) + '_cd.png'
        # output thumbnail unprocessed
        cd_thumb_image = Image.fromarray(np.array(cd_thumb, dtype=np.uint8))
        cd_thumb_image.save(thumbs_dir / filename_cd)
        # output thumbnail with background removed
        cd_filthumb_image = Image.fromarray(np.array(cd_fil_thumb, dtype=np.uint8))
        cd_filthumb_image.save(backgr_dir / filename_cd)
        # output pathologist annotations
        cd_annot_image = Image.fromarray(np.array(cd_annots*100, dtype=np.uint8))
        cd_annot_image.save(annot_out_dir / filename_cd)
        # output stuff removed by blood and mucus detector
        cd_bloodm_image = Image.fromarray(np.array(cd_bloodm, dtype=np.uint8))
        cd_bloodm_image.save(bloodm_dir / filename_cd)
        # output remaining tissue
        cd_tissue_image = Image.fromarray(np.array(cd_tissue, dtype=np.uint8))
        cd_tissue_image.save(tissue_dir / filename_cd)
        # output predicted labels
        cd_preds_image = Image.fromarray(np.array(cd_preds*100, dtype=np.uint8))
        cd_preds_image.save(labels_dir / filename_cd)

        # create and output mosaic of outputs
        top_row_cd = get_concat_h(get_concat_h(cd_thumb_image, cd_filthumb_image), cd_annot_image)
        low_row_cd = get_concat_h(get_concat_h(cd_bloodm_image, cd_tissue_image), cd_preds_image)
        get_concat_v(top_row_cd, low_row_cd).save(mosaic_dir / filename_cd)

    # sum all rows to get total for all slides
    cm_3class_all = np.sum(confusion_matrix_3class, axis=0)
    save_confusion_mat(cm_3class_all, experiment_root, experiment_name)
    cm_2class_all = np.sum(confusion_matrix_2class, axis=0)
    save_confusion_mat(cm_2class_all, experiment_root, experiment_name)
    





