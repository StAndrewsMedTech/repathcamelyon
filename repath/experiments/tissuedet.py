
import repath.data.datasets.tissue as tissue
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.tissue_detection import TissueDetector, TissueDetectorGreyScale, TissueDetectorAll, TissueDetectorOTSU
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
    tissue_patches_labelled = CombinedIndex.for_slide_indexes([tissue_patchsets_labelled])

    # create blank slides with just tissue detector labels
    tissue_patchsets_detected = SlidesIndex.index_dataset_blank(tissue_dataset, tissue_detector_test, patch_finder)
    tissue_patches_detected = CombinedIndex.for_slide_indexes([tissue_patchsets_detected])

    TP = np.sum(np.logical_and(tissue_patches_detected.patches_df.label, tissue_patches_labelled.patches_df.label))
    FP = np.sum(np.logical_and(tissue_patches_detected.patches_df.label, np.logical_not(tissue_patches_labelled.patches_df.label)))
    FN = np.sum(np.logical_and(np.logical_not(tissue_patches_detected.patches_df.label), tissue_patches_labelled.patches_df.label))
    TN = np.sum(np.logical_and(np.logical_not(tissue_patches_detected.patches_df.label), np.logical_not(tissue_patches_labelled.patches_df.label)))
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP) 

    print(f'Accuracy: {round(accuracy, 5)}, Recall: {round(recall, 5)}, Precision: {round(precision, 5)}')
    print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')

    heatmap_title = "Tissue Detector " + label + " - Accuracy :" + str(round(accuracy, 5))
    cm_out = conf_mat_plot_heatmap(np.array([TN, FP, FN, TP], dtype=np.int), ['background', 'foreground'], heatmap_title)
    output_name = label + '_confidence_matrix.png'
    cm_out.get_figure().savefig(experiment_root / output_name)


def greyscale() -> None:
    tissue_detector_test = TissueDetectorGreyScale()
    generic(tissue_detector_test, "Greyscale", 8)

