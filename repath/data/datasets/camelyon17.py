import os
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pdi
from os import walk
from os.path import join
from repath.data.annotations import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve


class Camelyon17(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)

    def load_annotations(self, file: Path) -> AnnotationSet:
        # if there is no annotation file the just pass and empty list
        annotations = load_annotations(file) if file else []
        labels_order = ["background", "tumor", "normal"]
        return AnnotationSet(annotations, self.labels, labels_order, "normal")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}

    def slide_label(self) -> Dict[str, int]:
        return {"negative": 0, "micro": 1, "macro":2, 'itc': 3}

def training():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon17" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    
    # all paths are relative to the dataset 'root'
    annotation_paths = sorted([join(root,f) for root,dirs,files in os.walk(root) for f in files if ".xml" in f])
    slide_paths = sorted([join(root,f) for root,dirs,files in os.walk(root) for f in files if ".tif" in f])
   
    slide_names = []
    for path in annotation_paths:
        head, tail = os.path.split(path)
        slide_name = tail.split('.')[0] + '.tif'
        slide_names.appned(slide_name)
    
    slides_with_annotations_paths = [path for path in slide_paths for slide_name in slide_names if slide_name in path]
    slides_without_annotations = [path for path  in slide_paths if path not in slides_with_annotations_paths]

    #slide level labels
    slide_labels = pd.read_csv(root /'stage_labels.csv')
    negative_slides = slide_labels[slide_labels.stage == 'negative']
    micro_slides = slide_labels[slide_labels.stage == 'micro']
    macro_slides = slide_labels[slide_labels.stage == 'macro']
    itc_slides = slide_labels[slide_labels.stage == 'itc']

    negative_slide_paths = [path for path in slide_paths for slide in negative_slides.patient if slide in path]
    micro_slide_paths = [path for path in slide_paths for slide in micro_slides.patient if slide in path]
    macro_slide_paths = [path for path in slide_paths for slide in macro_slides.patient if slide in path]
    itc_slide_paths = [path for path in slide_paths for slide in itc_slides.patient if slide in path]

    #patient level labels
    pN0_slides = slide_labels.loc[slide_labels['stage'].isin(['pN0'])]
    pN1_slides = slide_labels.loc[slide_labels['stage'].isin(['pN1'])]
    pN0(i+)_slides = slide_labels.loc[slide_labels['stage'].isin(['pN0(i+)'])]
    pN1mi_slides = slide_labels.loc[slide_labels['stage'].isin(['pN1mi'])]
    
    pN0_names = pN0_slides.patient.str.split('.').str[0]
    pN0_paths = [path for path in slide_paths for name in pN0_names if name in path]

    pN1_names = pN1_slides.patient.str.split('.').str[0]
    pN1_paths = [path for path in slide_paths for name in pN1_names if name in path]
    
    pN0(i+)_names = pN0(i+)_slides.patient.str.split('.').str[0]
    pN0(i+)_paths = [path for path in slide_paths for name in pN0(i+)_names if name in path]
    
    pN1mi_names = pN1mi_slides.patient.str.split('.').str[0]
    pN1mi_paths = [path for path in slide_paths for name in pN1mi_names if name in path]

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths 
    df["annotation"] = slide_with_annotation_paths + ["" for _ in range(len(slides_without_annotations_paths))]
    df["label"] = ['negative'] * len(negative_slide_paths) + ['macro'] * len(macro_slide_paths) + ['micro'] * len(micro_slide_paths) + ['itc'] * len(itc_slide_paths)
    df["patient_label"] = ['pN0'] * len(pN0_paths) + ['pN1']* len(pN1_paths) + ['pN0(i+)'] * len(pN0(i+)_paths) + ['pN1mi'] * len(pN1mi_paths)
    df["tags"] = ""

    return Camelyon17(root, df)



def testing():
    root = project_root() / "data" / "camelyon17" / "raw" / "testing"
    test_slides_dir = root / "patients"
    slide_paths = sorted([p.relative_to(root) for p in test_slides.dir.glob("*.tif")])
    slide_labels = pd.read_csv(root /'evaluation/submission_example.csv')
    negative_slides = slide_labels[slide_labels.stage == 'negative']
    micro_slides = slide_labels[slide_labels.stage == 'micro']
    macro_slides = slide_labels[slide_labels.stage == 'macro']
    itc_slides = slide_labels[slide_labels.stage == 'itc']

    negative_slide_paths = [path for path in slide_paths for slide in negative_slides.patient if slide in path]
    micro_slide_paths = [path for path in slide_paths for slide in micro_slides.patient if slide in path]
    macro_slide_paths = [path for path in slide_paths for slide in macro_slides.patient if slide in path]
    itc_slide_paths = [path for path in slide_paths for slide in itc_slides.patient if slide in path]

    #patient level labels
    pN0_slides = slide_labels.loc[slide_labels['stage'].isin(['pN0'])]
    pN1_slides = slide_labels.loc[slide_labels['stage'].isin(['pN1'])]
    pN0(i+)_slides = slide_labels.loc[slide_labels['stage'].isin(['pN0(i+)'])]
    pN1mi_slides = slide_labels.loc[slide_labels['stage'].isin(['pN1mi'])]

    pN0_names = pN0_slides.patient.str.split('.').str[0]
    pN0_paths = [path for path in slide_paths for name in pN0_names if name in path]

    pN1_names = pN1_slides.patient.str.split('.').str[0]
    pN1_paths = [path for path in slide_paths for name in pN1_names if name in path]

    pN0(i+)_names = pN0(i+)_slides.patient.str.split('.').str[0]
    pN0(i+)_paths = [path for path in slide_paths for name in pN0(i+)_names if name in path]

    pN1mi_names = pN1mi_slides.patient.str.split('.').str[0]
    pN1mi_paths = [path for path in slide_paths for name in pN1mi_names if name in path]

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["label"] = ['negative'] * len(negative_slide_paths) + ['macro'] * len(macro_slide_paths) + ['micro'] * len(micro_slide_paths) + ['itc'] * len(itc_slide_paths)
    df["patient_label"] = ['pN0'] * len(pN0_paths) + ['pN1']* len(pN1_paths) + ['pN0(i+)'] * len(pN0(i+)_paths) + ['pN1mi'] * len(pN1mi_paths)
    df["tags"] = ""


