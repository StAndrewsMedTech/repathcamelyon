import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from repath.data.datasets.dataset import Dataset
import repath.data.datasets.tissue as tissue
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.tissue_detection import TissueDetectorGreyScale, TissueDetectorOTSU, SimpleClosingTransform
from repath.preprocess.tissue_detection.tissue_metrics import calc_tissue_conf_mat, get_output_images, write_contours_to_file
from repath.utils.paths import project_root
from repath.utils.seeds import set_seed

"""
Global stuff
"""
experiment_name = "tissue"
experiment_root = project_root() / "experiments" / experiment_name

global_seed = 123


def greyscale() -> None:
    set_seed(global_seed)
    tissue_detector_test = TissueDetectorGreyScale()
    calc_tissue_conf_mat(tissue_detector_test, "greyscale", 7, experiment_root)
    get_output_images(tissue_detector_test, "greyscale", 7, experiment_root)
    write_contours_to_file(tissue_detector_test, "greyscale", 7, experiment_root, level_out=0)  


def otsu() -> None:
    set_seed(global_seed)
    tissue_detector_test = TissueDetectorOTSU()
    calc_tissue_conf_mat(tissue_detector_test, "otsu", 7, experiment_root)
    get_output_images(tissue_detector_test, "otsu", 7, experiment_root)
    write_contours_to_file(tissue_detector_test, "otsu", 7, experiment_root, level_out=0)


def greyscale_closing() -> None:
    set_seed(global_seed)
    morphology_transform = SimpleClosingTransform()
    tissue_detector_test = TissueDetectorGreyScale(morphology_transform)
    calc_tissue_conf_mat(tissue_detector_test, "greyscale_with_closing", 7, experiment_root)
    get_output_images(tissue_detector_test, "greyscale_with_closing", 7, experiment_root)
    write_contours_to_file(tissue_detector_test, "greyscale_with_closing", 7, experiment_root, level_out=0)


