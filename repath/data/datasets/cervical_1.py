from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pd
from repath.data.annotations import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root
from repath.utils.metrics import conf_mat_raw, plotROC, plotROCCI, pre_re_curve

class Cervical_subCategories(Cervical):
     def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"cin1": 0, "hpv":1 , "cin2": 2, "cin3": 3, "squamous_carcinoma": 4, "adenocarcinoma": 5, "cgin": 6, "other": 7 , "normal_inflammation": 8}


class Cervical(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path) -> AnnotationSet:
        group_labels = {"low_grade": "low_grade", "high_grade": "high_grade", "malignant": "malignant", "Normal/inflammation": "normal_inflammation"}
        annotations = load_annotations(file, group_labels) if file else []
        labels_order = ["low_grade",  "high_grade",  "malignant", "normal_inflammation"]
        return AnnotationSet(annotations, self.labels, labels_order, "low_grade")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"background" : 0, "low_grade": 1 , "high_grade": 2 ,  "malignant": 3, "normal_inflammation": 4}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"low_grade": 1 , "high_grade": 2 ,  "malignant": 3, "normal_inflammation": 4}

def training():
     """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "cervical" / "raw" / "training" 
    annotations_dir = root / "annotations"
    
    #slide directories    
    cin1_slide_dir = root / "low_grade" / "cin1"
    hpv_slide_dir = root / "low_grade" / "hpv"
    cin2_slide_dir = root / "high_grade" / "cin2"
    cin3_slide_dir = root / "high_grade" / "cin3"
    cgin_slide_dir = root / "malignant" / "cgin"
    adenocarcinoma_slide_dir = root / "malignant" / "adenocarcinoma"
    squamouscarcinoma_slide_dir =  root / "malignant" / "squamous_carcinoma"
    other_slide_dir = root / "malignant" / "other"
    Normal_slide_dir = root / "normal_inflammation"

    #annotation directories
    cin1_anno_dir = annotations_dir / "low_grade" / "cin1"
    hpv_anno_dir = annotations_dir / "low_grade" / "hpv"
    cin2_anno_dir = annotations_dir / "high_grade" / "cin2"
    cin3_anno_dir = annotations_dir / "high_grade" / "cin3"
    cgin_anno_dir = annotations_dir / "malignant" / "cgin"
    adenocarcinoma_anno_dir = annotations_dir / "malignant" / "adenocarcinoma"
    squamouscarcinoma_anno_dir =  annotations_dir / "malignant" / "squamous_carcinoma"
    other_anno_dir = annotations_dir / "malignant "/ "other"
    normal_anno_dir = annotations_dir / "normal_inflammation"



    #slide paths
    cin1_slide_paths = sorted([p.relative_to(root) for p in cin1_slide_dir.glob("*.isyntax")])
    hpv_slide_paths = sorted([p.relative_to(root) for p in hpv_slide_dir.glob("*.isyntax")])
    cin2_slide_paths = sorted([p.relative_to(root) for p in cin2_slide_dir.glob("*.isyntax")])
    cin3_slide_paths = sorted([p.relative_to(root) for p in cin3_slide_dir.glob("*.isyntax")])
    cgin_slide_paths = sorted([p.relative_to(root) for p in cgin_slide_dir.glob("*.isyntax")])
    adeno_slide_paths = sorted([p.relative_to(root) for p in adenocarcinoma_slide_dir.glob("*.isyntax")])
    squamou_slide_paths = sorted([p.relative_to(root) for p in squamouscarcinoma_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in other_slide_dir.glob("*.isyntax")])
    normal_slide_paths = sorted([p.relative_to(root) for p in Normal_slide_dir.glob("*.isyntax")])

    #annotation paths
    cin1_anno_paths = sorted([p.relative_to(root) for p in cin1_anno_dir.glob("*.txt")])
    hpv_anno_paths = sorted([p.relative_to(root) for p in hpv_anno_dir.glob("*.txt")])
    cin2_anno_paths = sorted([p.relative_to(root) for p in cin2_anno_dir.glob("*.txt")])
    cin3_anno_paths = sorted([p.relative_to(root) for p in cin3_anno_dir.glob("*.txt")])
    cgin_anno_paths = sorted([p.relative_to(root) for p in cgin_anno_dir.glob("*.txt")])
    adeno_anno_paths = sorted([p.relative_to(root) for p in adenocarcinoma_annodir.glob("*.txt")])
    squamou_anno_paths = sorted([p.relative_to(root) for p in squamouscarcinoma_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in other_anno_dir.glob("*.txt")])
    normal_anno_paths = sorted([p.relative_to(root) for p in normal_anno_dir.glob("*.txt")])


    df = pd.DataFrame()
    df["slide"] = cin1_slide_paths + hpv_slide_paths + cin2_slide_paths + cin3_slide_paths + cgin_slide_dir_paths + adeno_slide_paths +  squamou_slide_paths + other_slide_paths + Normal_slide_paths
    
    df["annotations"] = cin1_anno_paths + hpv_anno_paths + cin2_anno_paths + cin3_anno_paths + cgin_anno_dir_paths + adeno_anno_paths +  squamou_anno_paths + other_anno_paths + normal_anno_paths

    df["label"] = ['low_grade'] * len(cin1_slide_paths) + ['low_grade'] * len(hpv_slide_paths) + ['high_grade'] * len(cin2_slide_paths) + ['high_grade'] * ['malignant'] * len(cin3_slide_paths) + \
                  ['malignant'] * len(cgin_slide_dir_paths) + ['malignant'] * len(adeno_slide_paths) + ['malignant'] * len(squamou_slide_paths) + ['malignant'] * len(other_slide_paths) + \
                  ['normal_inflammation'] * len(normal_slide_paths)

    df["tags"] = ['cin1'] * len(cin1_slide_paths) + ['hpv'] * len(hpv_slide_paths) + ['cin2'] * len(cin2_slide_paths) + ['cin3'] * len(cin3_slide_paths) + ['cgin'] * len(cgin_slide_dir_paths) + \
                 ['adenocarcinoma'] * len(adeno_slide_paths) + ['squamous_carcinoma'] * len(squamou_slide_paths) + ['other'] * len(other_slide_paths) + ['normal_inflammation'] * len(normal_slide_paths)

    return Cervical(root, df)



def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Test data frame
    """
    
     # set up the paths to the slides and annotations
    root = project_root() / "data" / "cervical" / "raw" / "testing"
    annotations_dir = root / "annotations"

    #slide directories    
    cin1_slide_dir = root / "low_grade" / "cin1"
    hpv_slide_dir = root / "low_grade" / "hpv"
    cin2_slide_dir = root / "high_grade" / "cin2"
    cin3_slide_dir = root / "high_grade" / "cin3"
    cgin_slide_dir = root / "malignant" / "cgin"
    adenocarcinoma_slide_dir = root / "malignant" / "adenocarcinoma"
    squamouscarcinoma_slide_dir =  root / "malignant" / "squamous_carcinoma"
    other_slide_dir = root / "malignant" / "other"
    Normal_slide_dir = root / "normal_inflammation"

    #annotation directories
    cin1_anno_dir = annotations_dir / "low_grade" / "cin1"
    hpv_anno_dir = annotations_dir / "low_grade" / "hpv"
    cin2_anno_dir = annotations_dir / "high_grade" / "cin2"
    cin3_anno_dir = annotations_dir / "high_grade" / "cin3"
    cgin_anno_dir = annotations_dir / "malignant" / "cgin"
    adenocarcinoma_anno_dir = annotations_dir / "malignant" / "adenocarcinoma"
    squamouscarcinoma_anno_dir =  annotations_dir / "malignant" / "squamous_carcinoma"
    other_anno_dir = annotations_dir / "malignant "/ "other"
    normal_anno_dir = annotations_dir / "normal_inflammation"


     #slide paths
    cin1_slide_paths = sorted([p.relative_to(root) for p in cin1_slide_dir.glob("*.isyntax")])
    hpv_slide_paths = sorted([p.relative_to(root) for p in hpv_slide_dir.glob("*.isyntax")])
    cin2_slide_paths = sorted([p.relative_to(root) for p in cin2_slide_dir.glob("*.isyntax")])
    cin3_slide_paths = sorted([p.relative_to(root) for p in cin3_slide_dir.glob("*.isyntax")])
    cgin_slide_paths = sorted([p.relative_to(root) for p in cgin_slide_dir.glob("*.isyntax")])
    adeno_slide_paths = sorted([p.relative_to(root) for p in adenocarcinoma_slide_dir.glob("*.isyntax")])
    squamou_slide_paths = sorted([p.relative_to(root) for p in squamouscarcinoma_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in other_slide_dir.glob("*.isyntax")])
    normal_slide_paths = sorted([p.relative_to(root) for p in Normal_slide_dir.glob("*.isyntax")])

    #annotation paths
    cin1_anno_paths = sorted([p.relative_to(root) for p in cin1_anno_dir.glob("*.txt")])
    hpv_anno_paths = sorted([p.relative_to(root) for p in hpv_anno_dir.glob("*.txt")])
    cin2_anno_paths = sorted([p.relative_to(root) for p in cin2_anno_dir.glob("*.txt")])
    cin3_anno_paths = sorted([p.relative_to(root) for p in cin3_anno_dir.glob("*.txt")])
    cgin_anno_paths = sorted([p.relative_to(root) for p in cgin_anno_dir.glob("*.txt")])
    adeno_anno_paths = sorted([p.relative_to(root) for p in adenocarcinoma_annodir.glob("*.txt")])
    squamou_anno_paths = sorted([p.relative_to(root) for p in squamouscarcinoma_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in other_anno_dir.glob("*.txt")])
    normal_anno_paths = sorted([p.relative_to(root) for p in normal_anno_dir.glob("*.txt")])


    df = pd.DataFrame()
    df["slide"] = cin1_slide_paths + hpv_slide_paths + cin2_slide_paths + cin3_slide_paths + cgin_slide_dir_paths + adeno_slide_paths +  squamou_slide_paths + other_slide_paths + Normal_slide_paths

    df["annotations"] = cin1_anno_paths + hpv_anno_paths + cin2_anno_paths + cin3_anno_paths + cgin_anno_dir_paths + adeno_anno_paths +  squamou_anno_paths + other_anno_paths + normal_anno_paths

    df["label"] = ['low_grade'] * len(cin1_slide_paths) + ['low_grade'] * len(hpv_slide_paths) + ['high_grade'] * len(cin2_slide_paths) + ['high_grade'] * ['malignant'] * len(cin3_slide_paths) + \
                  ['malignant'] * len(cgin_slide_dir_paths) + ['malignant'] * len(adeno_slide_paths) + ['malignant'] * len(squamou_slide_paths) + ['malignant'] * len(other_slide_paths) + \
                  ['normal_inflammation'] * len(normal_slide_paths)

    df["tags"] = ['cin1'] * len(cin1_slide_paths) + ['hpv'] * len(hpv_slide_paths) + ['cin2'] * len(cin2_slide_paths) + ['cin3'] * len(cin3_slide_paths) + ['cgin'] * len(cgin_slide_dir_paths) + \
                 ['adenocarcinoma'] * len(adeno_slide_paths) + ['squamous_carcinoma'] * len(squamou_slide_paths) + ['other'] * len(other_slide_paths) + ['normal_inflammation'] * len(normal_slide_paths)


    return Cervical(root, df)




