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
        return {"CIN1": 0, "HPV":1 , "CIN2": 2, "CIN3": 3, "Squamous carcinoma": 4, "Adenocarcinoma": 5, "CGIN": 6, "Other": 7 , "Normal/inflammation": 8}


class Cervical(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path) -> AnnotationSet:
        group_labels = {"Low Grade": "low_grade", "High Grade": "high_grade", "Malignant": "malignant", "Normal/inflammation": "normal_inflammation"}
        annotations = load_annotations(file, group_labels) if file else []
        labels_order = ["low_grade",  "high_grade",  "malignant", "normal_inflammation"]
        return AnnotationSet(annotations, self.labels, labels_order, "low_grade")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"Background" : 0, "Low Grade": 1 , "High Grade": 2 ,  "Malignant": 3, "Normal/inflammation": 4}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"Low Grade": 1 , "High Grade": 2 ,  "Malignant": 3, "Normal/inflammation": 4}

def training():
     """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "cervical" / "raw" / "training" 
    annotations_dir = root / "annotations"
    
    #slide directories    
    CIN1_slide_dir = root / "Low Grade" / "CIN1"
    HPV_slide_dir = root / "Low Grade" / "HPV"
    CIN2_slide_dir = root / "High Grade" / "CIN2"
    CIN3_slide_dir = root / "High Grade" / "CIN3"
    CGIN_slide_dir = root / "Malignant" / "CGIN"
    Adenocarcinoma_slide_dir = root / "Malignant" / "Adenocarcinoma"
    Squamouscarcinoma_slide_dir =  root / "Malignant" / "Squamous carcinoma"
    Other_slide_dir = root / "Malignant" / "Other"
    Normal_slide_dir = root / "Normal-inflammation"

    #annotation directories
    CIN1_anno_dir = annotations_dir / "Low Grade" / "CIN1"
    HPV_anno_dir = annotations_dir / "Low Grade" / "HPV"
    CIN2_anno_dir = annotations_dir / "High Grade" / "CIN2"
    CIN3_anno_dir = annotations_dir / "High Grade" / "CIN3"
    CGIN_anno_dir = annotations_dir / "Malignant" / "CGIN"
    Adenocarcinoma_anno_dir = annotations_dir / "Malignant" / "Adenocarcinoma"
    Squamouscarcinoma_anno_dir =  annotations_dir / "Malignant" / "Squamous carcinoma"
    Other_anno_dir = annotations_dir / "Malignant "/ "Other"
    Normal_anno_dir = annotations_dir / "Normal-imflammation"



    #slide paths
    cin1_slide_paths = sorted([p.relative_to(root) for p in CIN1_slide_dir.glob("*.isyntax")])
    hpv_slide_paths = sorted([p.relative_to(root) for p in HPV_slide_dir.glob("*.isyntax")])
    cin2_slide_paths = sorted([p.relative_to(root) for p in CIN2_slide_dir.glob("*.isyntax")])
    cin3_slide_paths = sorted([p.relative_to(root) for p in CIN3_slide_dir.glob("*.isyntax")])
    cgin_slide_paths = sorted([p.relative_to(root) for p in CGIN_slide_dir.glob("*.isyntax")])
    adeno_slide_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_slide_dir.glob("*.isyntax")])
    squamou_slide_paths = sorted([p.relative_to(root) for p in Squamouscarcinoma_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in Other_slide_dir.glob("*.isyntax")])
    normal_slide_paths = sorted([p.relative_to(root) for p in Normal_slide_dir.glob("*.isyntax")])

    #annotation paths
    cin1_anno_paths = sorted([p.relative_to(root) for p in CIN1_anno_dir.glob("*.txt")])
    hpv_anno_paths = sorted([p.relative_to(root) for p in HPV_anno_dir.glob("*.txt")])
    cin2_anno_paths = sorted([p.relative_to(root) for p in CIN2_anno_dir.glob("*.txt")])
    cin3_anno_paths = sorted([p.relative_to(root) for p in CIN3_anno_dir.glob("*.txt")])
    cgin_anno_paths = sorted([p.relative_to(root) for p in CGIN_anno_dir.glob("*.txt")])
    adeno_anno_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_annodir.glob("*.txt")])
    squamou_anno_paths = sorted([p.relative_to(root) for p in Squamouscarcinoma_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in Other_anno_dir.glob("*.txt")])
    normal_anno_paths = sorted([p.relative_to(root) for p in Normal_anno_dir.glob("*.txt")])


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
    CIN1_slide_dir = root / "Low Grade" / "CIN1"
    HPV_slide_dir = root / "Low Grade" / "HPV"
    CIN2_slide_dir = root / "High Grade" / "CIN2"
    CIN3_slide_dir = root / "High Grade" / "CIN3"
    CGIN_slide_dir = root / "Malignant" / "CGIN"
    Adenocarcinoma_slide_dir = root / "Malignant" / "Adenocarcinoma"
    Squamouscarcinoma_slide_dir =  root / "Malignant" / "Squamous carcinoma"
    Other_slide_dir = root / "Malignant" / "Other"
    Normal_slide_dir = root / "Normal-inflammation"

    #annotation directories
    CIN1_anno_dir = annotations_dir / "Low Grade" / "CIN1"
    HPV_anno_dir = annotations_dir / "Low Grade" / "HPV"
    CIN2_anno_dir = annotations_dir / "High Grade" / "CIN2"
    CIN3_anno_dir = annotations_dir / "High Grade" / "CIN3"
    CGIN_anno_dir = annotations_dir / "Malignant" / "CGIN"
    Adenocarcinoma_anno_dir = annotations_dir / "Malignant" / "Adenocarcinoma"
    Squamouscarcinoma_anno_dir =  annotations_dir / "Malignant" / "Squamous carcinoma"
    Other_anno_dir = annotations_dir / "Malignant "/ "Other"
    Normal_anno_dir = annotations_dir / "Normal-imflammation"


     #slide paths
    cin1_slide_paths = sorted([p.relative_to(root) for p in CIN1_slide_dir.glob("*.isyntax")])
    hpv_slide_paths = sorted([p.relative_to(root) for p in HPV_slide_dir.glob("*.isyntax")])
    cin2_slide_paths = sorted([p.relative_to(root) for p in CIN2_slide_dir.glob("*.isyntax")])
    cin3_slide_paths = sorted([p.relative_to(root) for p in CIN3_slide_dir.glob("*.isyntax")])
    cgin_slide_paths = sorted([p.relative_to(root) for p in CGIN_slide_dir.glob("*.isyntax")])
    adeno_slide_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_slide_dir.glob("*.isyntax")])
    squamou_slide_paths = sorted([p.relative_to(root) for p in Squamouscarcinoma_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in Other_slide_dir.glob("*.isyntax")])
    normal_slide_paths = sorted([p.relative_to(root) for p in Normal_slide_dir.glob("*.isyntax")])

    #annotation paths
    cin1_anno_paths = sorted([p.relative_to(root) for p in CIN1_anno_dir.glob("*.txt")])
    hpv_anno_paths = sorted([p.relative_to(root) for p in HPV_anno_dir.glob("*.txt")])
    cin2_anno_paths = sorted([p.relative_to(root) for p in CIN2_anno_dir.glob("*.txt")])
    cin3_anno_paths = sorted([p.relative_to(root) for p in CIN3_anno_dir.glob("*.txt")])
    cgin_anno_paths = sorted([p.relative_to(root) for p in CGIN_anno_dir.glob("*.txt")])
    adeno_anno_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_annodir.glob("*.txt")])
    squamou_anno_paths = sorted([p.relative_to(root) for p in Squamouscarcinoma_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in Other_anno_dir.glob("*.txt")])
    normal_anno_paths = sorted([p.relative_to(root) for p in Normal_anno_dir.glob("*.txt")])


    df = pd.DataFrame()
    df["slide"] = cin1_slide_paths + hpv_slide_paths + cin2_slide_paths + cin3_slide_paths + cgin_slide_dir_paths + adeno_slide_paths +  squamou_slide_paths + other_slide_paths + Normal_slide_paths

    df["annotations"] = cin1_anno_paths + hpv_anno_paths + cin2_anno_paths + cin3_anno_paths + cgin_anno_dir_paths + adeno_anno_paths +  squamou_anno_paths + other_anno_paths + normal_anno_paths

    df["label"] = ['low_grade'] * len(cin1_slide_paths) + ['low_grade'] * len(hpv_slide_paths) + ['high_grade'] * len(cin2_slide_paths) + ['high_grade'] * ['malignant'] * len(cin3_slide_paths) + \
                  ['malignant'] * len(cgin_slide_dir_paths) + ['malignant'] * len(adeno_slide_paths) + ['malignant'] * len(squamou_slide_paths) + ['malignant'] * len(other_slide_paths) + \
                  ['normal_inflammation'] * len(normal_slide_paths)

    df["tags"] = ['cin1'] * len(cin1_slide_paths) + ['hpv'] * len(hpv_slide_paths) + ['cin2'] * len(cin2_slide_paths) + ['cin3'] * len(cin3_slide_paths) + ['cgin'] * len(cgin_slide_dir_paths) + \
                 ['adenocarcinoma'] * len(adeno_slide_paths) + ['squamous_carcinoma'] * len(squamou_slide_paths) + ['other'] * len(other_slide_paths) + ['normal_inflammation'] * len(normal_slide_paths)


    return Cervical(root, df)




