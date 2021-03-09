from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import repath.data.datasets.tissue as tissue
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.tissue_detection import TissueDetector, TissueDetectorGreyScale, TissueDetectorAll, TissueDetectorOTSU, SimpleClosingTransform
from repath.utils.metrics import conf_mat_plot_heatmap
from repath.utils.paths import project_root
from repath.utils.seeds import set_seed

"""
Global stuff
"""
experiment_name = "tissue"
experiment_root = project_root() / "experiments" / experiment_name

global_seed = 123


def generic(tissue_detector_test: TissueDetector, label: str, level_in: int) -> None:
    set_seed(global_seed)
    tissue_dataset = tissue.tissue()
    psize = 2 ** level_in
    patch_finder = GridPatchFinder(labels_level=level_in, patch_level=0, patch_size=psize, stride=psize, remove_background=False)
    # The tissue detector is applied over the top of the patch finder
    # tissue detector all classes everything as foreground so will not change the foreground background labels from the annotations
    tissue_detector_all = TissueDetectorAll()
    tissue_patchsets_labelled = SlidesIndex.index_dataset(tissue_dataset, tissue_detector_all, patch_finder)
    truth_dir = experiment_root / label / 'truth'
    tissue_patchsets_labelled.save(truth_dir)
    tissue_patches_labelled = CombinedIndex.for_slide_indexes([tissue_patchsets_labelled])
    

    # create blank slides with just tissue detector labels
    tissue_patchsets_detected = SlidesIndex.index_dataset_blank(tissue_dataset, tissue_detector_test, patch_finder)
    detection_dir = experiment_root / label / 'tissue_detection'
    tissue_patchsets_detected.save(detection_dir)
    tissue_patches_detected = CombinedIndex.for_slide_indexes([tissue_patchsets_detected])


    TP = np.sum(np.logical_and(tissue_patches_detected.patches_df.label, tissue_patches_labelled.patches_df.label))
    FP = np.sum(np.logical_and(tissue_patches_detected.patches_df.label, np.logical_not(tissue_patches_labelled.patches_df.label)))
    FN = np.sum(np.logical_and(np.logical_not(tissue_patches_detected.patches_df.label), tissue_patches_labelled.patches_df.label))
    TN = np.sum(np.logical_and(np.logical_not(tissue_patches_detected.patches_df.label), np.logical_not(tissue_patches_labelled.patches_df.label)))
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP) 
    confmat = np.array([[TN, FP], [FN, TP]], dtype=np.int)


    print(f'Accuracy: {round(accuracy, 5)}, Recall: {round(recall, 5)}, Precision: {round(precision, 5)}')
    print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')

    heatmap_title = "Tissue Detector " + label + " - Accuracy :" + str(round(accuracy, 5))
    cm_out = conf_mat_plot_heatmap(confmat, ['background', 'foreground'], heatmap_title)
    output_name = label + '_confidence_matrix.png'
    cm_out.get_figure().savefig(experiment_root / output_name)

    save_output_images(truth_dir, detection_dir, tissue_dataset, patch_finder.labels_level, label)


def get_slide_output_arrays(true_patches, detected_patches, patch_size):
    def to_thumbnail_tissue(df_in, col_name: str) -> np.array:
        max_rows = int(np.max(df_in.row)) + 1
        max_cols = int(np.max(df_in.col)) + 1

        # create a blank thumbnail
        thumbnail_out = np.zeros((max_rows, max_cols))
        df_out = df_in[df_in[col_name] > 0]

        # for each row in dataframe set the value of the pixel specified by row and column to the probability in clazz
        for rw in range(df_out.shape[0]):
            df_row = df_out.iloc[rw]
            thumbnail_out[int(df_row.row), int(df_row.col)] = df_row[col_name]

        return thumbnail_out
    
    # change x y from zero level positions to pthumbnail level size
    true_patches['col'] = true_patches.x.divide(patch_size)
    true_patches['row'] = true_patches.y.divide(patch_size)
    detected_patches['col'] = detected_patches.x.divide(patch_size)
    detected_patches['row'] = detected_patches.y.divide(patch_size)
    
    # create thumbnails from the patch indexes
    tiss_np = np.array(to_thumbnail_tissue(true_patches, 'label'), dtype=bool)
    dets_np = np.array(to_thumbnail_tissue(detected_patches, 'label'), dtype=bool)
    
    # work out tp, fp etc arrays
    tp_arr = np.expand_dims(np.logical_and(tiss_np, dets_np), axis=-1)
    fp_arr = np.expand_dims(np.logical_and(np.logical_not(tiss_np), dets_np), axis=-1)
    fn_arr = np.expand_dims(np.logical_and(np.logical_not(dets_np), tiss_np), axis=-1)
    tn_arr = np.expand_dims(np.logical_and(np.logical_not(tiss_np), np.logical_not(dets_np)), axis=-1)

    return tn_arr, fp_arr, fn_arr, tp_arr


def create_colour_output(tn_arr, fp_arr, fn_arr, tp_arr):
    
    # create a coloured ouput array
    color_result = np.zeros((tp_arr.shape[0], tp_arr.shape[1], 3))
    color_result = np.where(tp_arr, [255, 255, 255], color_result)
    color_result = np.where(fp_arr, [0, 0, 255], color_result)
    color_result = np.where(fn_arr, [255, 0, 0], color_result)
    color_result = np.where(tn_arr, [0, 0, 0], color_result)
    color_im = Image.fromarray(np.array(color_result, dtype=np.uint8))
    
    return color_im


def create_overlay_output(slide_path, dataset, labels_level, fp_arr, fn_arr):
    with dataset.slide_cls(slide_path) as slide:
        thumb = slide.get_thumbnail(labels_level)

    # change pure black to pure white
    thumbr = thumb[:, :, 0] == 0
    thumbg = thumb[:, :, 1] == 0
    thumbb = thumb[:, :, 2] == 0
    thumb_mask = np.expand_dims(np.logical_and(np.logical_and(thumbr, thumbg), thumbb), axis=-1)
    thumb = np.where(thumb_mask, [255,255,255], thumb)
    thumb = np.array(thumb, dtype=np.uint8)

    # create overlay mask
    color_result = np.zeros((fp_arr.shape[0], fp_arr.shape[1], 3))
    color_result = np.where(fp_arr, [0, 0, 1], color_result)
    color_result = np.where(fn_arr, [1, 0, 0], color_result)

    # composite mask over thumbnail
    thumb_hsv = color.rgb2hsv(image)
    color_mask_hsv = color.rgb2hsv(color_result)

    thumb_hsv[..., 0] = color_mask_hsv[..., 0]
    thumb_hsv[..., 1] += color_mask_hsv[..., 1] * 0.7
    img_masked = color.hsv2rgb(thumb_hsv)

    overlay_im = Image.fromarray(np.array(img_masked*255, dtype=np.uint8))
    return overlay_im


def save_output_images(truth_dir, detection_dir, tissue_dataset, labels_level, label):
    truth = SlidesIndex.load(tissue_dataset, truth_dir)
    detection = SlidesIndex.load(tissue_dataset, detection_dir)
    outdir_colour = experiment_root / label / 'colour_output_images'
    outdir_overlay = experiment_root / label / 'overlay_output_images'
    outdir_colour.mkdir(parents=True, exist_ok=True)
    outdir_overlay.mkdir(parents=True, exist_ok=True)
    for idx in range(len(truth)):
        true_patches = truth[idx].patches_df
        true_path = truth[idx].slide_path
        print(true_path)
        detection_patches = detection[idx].patches_df
        patch_size = detection[idx].patch_size
        tn_arr, fp_arr, fn_arr, tp_arr = get_slide_output_arrays(true_patches, detection_patches, patch_size)
        colour_thumb = create_colour_output(tn_arr, fp_arr, fn_arr, tp_arr)
        outpath_colour = outdir_colour / str(Path(true_path).stem + '.jpg')
        colour_thumb.save(outpath_colour)
        slide_path = detection[idx].slide_path
        #overlay_thumb = create_overlay_output(slide_path, detection.dataset, labels_level, fp_arr, fn_arr)
        #outpath_overlay = outdir_overlay / str(Path(true_path).stem + '.jpg')
        #overlay_thumb.save(outpath_thumb)


def greyscale() -> None:
    tissue_detector_test = TissueDetectorGreyScale()
    generic(tissue_detector_test, "Greyscale", 7)


def otsu() -> None:
    tissue_detector_test = TissueDetectorOTSU()
    generic(tissue_detector_test, "Otsu", 7)


def greyscale_closing() -> None:
    morphology_transform = SimpleClosingTransform()
    tissue_detector_test = TissueDetectorGreyScale(morphology_transform)
    generic(tissue_detector_test, "Greyscale", 7)
