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

class Endometrial_subCategories(Endometrial):
     def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"Adenocarcinoma": 0, "Carcinosarcoma":1 , "Sarcoma": 2, "Hyperplasia with atypia": 3, "Other": 4, "Insufficient": 5, 
                "Proliferative": 6, "Secretory": 7 , "Menstrual": 8, "Innactive/atrophic": 9, "Hormonal": 10}

class Endometrial_subCategories(Endometrial):
     def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    @property
    def labels(self) -> Dict[str, int]:
        return {"Adenocarcinoma": 0, "Carcinosarcoma":1 , "Sarcoma": 2, "Hyperplasia with atypia": 3, "Other": 4, "Insufficient": 5, 
                "Proliferative": 6, "Secretory": 7 , "Menstrual": 8, "Innactive/atrophic": 9}, "Hormonal": 10}


class Endometrial(Dataset):
    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(root, paths)
    
    def load_annotations(self, file: Path) -> AnnotationSet:
        group_labels = {"Malignant": "malignant", "Insufficient": "insufficient", "Other/benign": "other_benign"}
        annotations = load_annotations(file, group_labels) if file else []
        labels_order = [ "malignant", "insufficient", "other_benign"]
        return AnnotationSet(annotations, self.labels, labels_order, "malignant")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    ## How to put sub category labels here?
    @property
    def labels(self) -> Dict[str, int]:
        return {"Background" : 0, "Malignant": 1 , "Insufficient": 2 ,  "Other/benign": 3}

    @property
    def slide_labels(self) -> Dict[str, int]:
        return {"Malignant": 1 , "Insufficient": 2 ,  "Other/benign": 3}

def training():
     """ Generated a data-frame of slide_path, annotation_path, label and tags for train dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "endometrial" / "raw" / "training" 
    annotations_dir = root / "annotations"
    
    #slide directories    
    Adenocarcinoma_slide_dir = root / "Malignant" / "Adenocarcinoma"
    Carcinosarcoma_slide_dir = root / "Malignant" / "Carcinosarcoma"
    Sarcoma_slide_dir = root / "Malignant" / "Sarcoma"
    Hyperplasia_slide_dir = root / "Malignant" / "Hyperplasia"
    Other_slide_dir = root / "Malignant" / "Other"
    Insufficient_slide_dir = root / "Insufficient" 
    Proliferative_slide_dir = root / "Other-benign" / "Proliferative"
    Secretory_slide_dir = root / "Other-benign" / "Secretory"
    Menstrual_slide_dir = root / "Other-benign" / "Manstrual"
    Innactive_slide_dir = root / "Other-benign" / "Innactiva"
    Hormonal_slide_dir = root / "Other-benign" / "Hormonal"
    
    #annotation directories
    Adenocarcinoma_anno_dir = annotations_dir / "Malignant" / "Adenocarcinoma"
    Carcinosarcoma_anno_dir = annotations_dir / "Malignant" / "Carcinosarcoma"
    Sarcoma_anno_dir = annotations_dir / "Malignant" / "Sarcoma"
    Hyperplasia_anno_dir = annotations_dir / "Malignant" / "Hyperplasia"
    Other_anno_dir = annotations_dir / "Malignant" / "Other"
    Insufficient_anno_dir = annotations_dir / "Insufficient" 
    Proliferative_anno_dir = annotations_dir / "Other-benign" / "Proliferative"
    Secretory_anno_dir = annotations_dir / "Other-benign" / "Secretory"
    Menstrual_anno_dir = annotations_dir / "Other-benign" / "Manstrual"
    Innactive_anno_dir = annotations_dir / "Other-benign" / "Innactiva"
    Hormonal_anno_dir = annotations_dir / "Other-benign" / "Hormonal"

    #slide paths
    adenocarcinoma_slide_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_slide_dir.glob("*.isyntax")])
    carcinosarcoma_slide_paths = sorted([p.relative_to(root) for p in Carcinosarcoma_slide_dir.glob("*.isyntax")])
    sarcoma_slide_paths = sorted([p.relative_to(root) for p in Sarcoma_slide_dir.glob("*.isyntax")])
    hyperplasia_slide_paths = sorted([p.relative_to(root) for p in Hyperplasia_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in Other_slide_dir.glob("*.isyntax")])
    insufficient_slide_paths = sorted([p.relative_to(root) for p in Insufficient_slide_dir.glob("*.isyntax")])
    proliferative_slide_paths = sorted([p.relative_to(root) for p in Proliferative_slide_dir.glob("*.isyntax")])
    secretory_slide_paths = sorted([p.relative_to(root) for p in Secretory_slide_dir.glob("*.isyntax")])
    menstrual_slide_paths = sorted([p.relative_to(root) for p in  Menstrual_slide_dir.glob("*.isyntax")])
    innactive_slide_paths = sorted([p.relative_to(root) for p in Innactive_slide_dir.glob("*.isyntax")])
    hormonal_slide_paths = sorted([p.relative_to(root) for p in Hormonal_slide_dir.glob("*.isyntax")])

    #annotation paths
    Adenocarcinoma_anno_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_anno_dir.glob("*.txt")])
    carcinosarcoma_anno_paths = sorted([p.relative_to(root) for p in Carcinosarcoma_anno_dir.glob("*.txt")])
    sarcoma_anno_paths = sorted([p.relative_to(root) for p in Sarcoma_anno_dir.glob("*.txt")])
    hyperplasia_anno_paths = sorted([p.relative_to(root) for p in Hyperplasia_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in Other_anno_dir.glob("*.txt")])
    insufficient_anno_paths = sorted([p.relative_to(root) for p in Insufficient_anno_dir.glob("*.txt")])
    proliferative_anno_paths = sorted([p.relative_to(root) for p in Proliferative_anno_dir.glob("*.txt")])
    secretory_anno_paths = sorted([p.relative_to(root) for p in Secretory_anno_dir.glob("*.txt")])
    menstrual_anno_paths = sorted([p.relative_to(root) for p in  Menstrual_anno_dir.glob("*.txt")])
    innactive_anno_paths = sorted([p.relative_to(root) for p in Innactive_anno_dir.glob("*.txt")])
    hormonal_anno_paths = sorted([p.relative_to(root) for p in Hormonal_anno_dir.glob("*.txt")])

    df = pd.DataFrame()
    df["slide"] = adenocarcinoma_slide_paths + carcinosarcoma_slide_paths + sarcoma_slide_paths + hyperplasia_slide_paths + other_slide_paths + insufficient_slide_paths + proliferative_slide_paths + \
                secretory_slide_paths +  menstrual_slide_paths + innactive_slide_paths + hormonal_slide_paths
    
    df["annotations"] = adenocarcinoma_anno_paths + carcinosarcoma_anno_paths + sarcoma_anno_paths + hyperplasia_anno_paths + other_anno_paths + insufficient_anno_paths + proliferative_anno_paths + \
                secretory_anno_paths +  menstrual_anno_paths + innactive_anno_paths + hormonal_anno_paths

    df["label"] = ['malignant'] * len(adenocarcinoma_slide_paths) + ['malignant'] * len(carcinosarcoma_slide_paths) + ['malignant'] * len(sarcoma_slide_paths) + ['malignant'] * len(hyperplasia_slide_paths) + ['malignant'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['other_benign'] * len(proliferative_slide_paths) + ['other_benign'] * len(secretory_slide_paths) +   ['other_benign'] * len(menstrual_slide_paths) +  ['other_benign'] * len(innactive_slide_paths) + ['other_benign'] * len(hormonal_slide_paths)



    df["tags"] = ['adenocarcinoma'] * len(adenocarcinoma_slide_paths) + ['carcinosarcoma'] * len(carcinosarcoma_slide_paths) + ['sarcoma'] * len(sarcoma_slide_paths) + ['hyperplasia'] * len(hyperplasia_slide_paths) + ['other'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['proliferative'] * len(proliferative_slide_paths) + ['secretory'] * len(secretory_slide_paths) +   ['menstrual'] * len(menstrual_slide_paths) +  ['innactive'] * len(innactive_slide_paths) + ['hormonal'] * len(hormonal_slide_paths)


    
    return Cervical(root, df)



def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Test data frame
    """
    
     # set up the paths to the slides and annotations
    root = project_root() / "data" / "endometrial" / "raw" / "testing"
    annotations_dir = root / "annotations"

    #slide directories    
    Adenocarcinoma_slide_dir = root / "Malignant" / "Adenocarcinoma"
    Carcinosarcoma_slide_dir = root / "Malignant" / "Carcinosarcoma"
    Sarcoma_slide_dir = root / "Malignant" / "Sarcoma"
    Hyperplasia_slide_dir = root / "Malignant" / "Hyperplasia"
    Other_slide_dir = root / "Malignant" / "Other"
    Insufficient_slide_dir = root / "Insufficient"
    Proliferative_slide_dir = root / "Other-benign" / "Proliferative"
    Secretory_slide_dir = root / "Other-benign" / "Secretory"
    Menstrual_slide_dir = root / "Other-benign" / "Manstrual"
    Innactive_slide_dir = root / "Other-benign" / "Innactiva"
    Hormonal_slide_dir = root / "Other-benign" / "Hormonal"

    #annotation directories
    Adenocarcinoma_anno_dir = annotations_dir / "Malignant" / "Adenocarcinoma"
    Carcinosarcoma_anno_dir = annotations_dir / "Malignant" / "Carcinosarcoma"
    Sarcoma_anno_dir = annotations_dir / "Malignant" / "Sarcoma"
    Hyperplasia_anno_dir = annotations_dir / "Malignant" / "Hyperplasia"
    Other_anno_dir = annotations_dir / "Malignant" / "Other"
    Insufficient_anno_dir = annotations_dir / "Insufficient"
    Proliferative_anno_dir = annotations_dir / "Other-benign" / "Proliferative"
    Secretory_anno_dir = annotations_dir / "Other-benign" / "Secretory"
    Menstrual_anno_dir = annotations_dir / "Other-benign" / "Manstrual"
    Innactive_anno_dir = annotations_dir / "Other-benign" / "Innactiva"
    Hormonal_anno_dir = annotations_dir / "Other-benign" / "Hormonal"

    #slide paths
    adenocarcinoma_slide_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_slide_dir.glob("*.isyntax")])
    carcinosarcoma_slide_paths = sorted([p.relative_to(root) for p in Carcinosarcoma_slide_dir.glob("*.isyntax")])
    sarcoma_slide_paths = sorted([p.relative_to(root) for p in Sarcoma_slide_dir.glob("*.isyntax")])
    hyperplasia_slide_paths = sorted([p.relative_to(root) for p in Hyperplasia_slide_dir.glob("*.isyntax")])
    other_slide_paths = sorted([p.relative_to(root) for p in Other_slide_dir.glob("*.isyntax")])
    insufficient_slide_paths = sorted([p.relative_to(root) for p in Insufficient_slide_dir.glob("*.isyntax")])
    proliferative_slide_paths = sorted([p.relative_to(root) for p in Proliferative_slide_dir.glob("*.isyntax")])
    secretory_slide_paths = sorted([p.relative_to(root) for p in Secretory_slide_dir.glob("*.isyntax")])
    menstrual_slide_paths = sorted([p.relative_to(root) for p in  Menstrual_slide_dir.glob("*.isyntax")])
    innactive_slide_paths = sorted([p.relative_to(root) for p in Innactive_slide_dir.glob("*.isyntax")])
    hormonal_slide_paths = sorted([p.relative_to(root) for p in Hormonal_slide_dir.glob("*.isyntax")])


    #annotation paths
    Adenocarcinoma_anno_paths = sorted([p.relative_to(root) for p in Adenocarcinoma_anno_dir.glob("*.txt")])
    carcinosarcoma_anno_paths = sorted([p.relative_to(root) for p in Carcinosarcoma_anno_dir.glob("*.txt")])
    sarcoma_anno_paths = sorted([p.relative_to(root) for p in Sarcoma_anno_dir.glob("*.txt")])
    hyperplasia_anno_paths = sorted([p.relative_to(root) for p in Hyperplasia_anno_dir.glob("*.txt")])
    other_anno_paths = sorted([p.relative_to(root) for p in Other_anno_dir.glob("*.txt")])
    insufficient_anno_paths = sorted([p.relative_to(root) for p in Insufficient_anno_dir.glob("*.txt")])
    proliferative_anno_paths = sorted([p.relative_to(root) for p in Proliferative_anno_dir.glob("*.txt")])
    secretory_anno_paths = sorted([p.relative_to(root) for p in Secretory_anno_dir.glob("*.txt")])
    menstrual_anno_paths = sorted([p.relative_to(root) for p in  Menstrual_anno_dir.glob("*.txt")])
    innactive_anno_paths = sorted([p.relative_to(root) for p in Innactive_anno_dir.glob("*.txt")])
    hormonal_anno_paths = sorted([p.relative_to(root) for p in Hormonal_anno_dir.glob("*.txt")])

    df = pd.DataFrame()
    df["slide"] = adenocarcinoma_slide_paths + carcinosarcoma_slide_paths + sarcoma_slide_paths + hyperplasia_slide_paths + other_slide_paths + insufficient_slide_paths + proliferative_slide_paths + \
                secretory_slide_paths +  menstrual_slide_paths + innactive_slide_paths + hormonal_slide_paths

    df["annotations"] = adenocarcinoma_anno_paths + carcinosarcoma_anno_paths + sarcoma_anno_paths + hyperplasia_anno_paths + other_anno_paths + insufficient_anno_paths + proliferative_anno_paths + \
                secretory_anno_paths +  menstrual_anno_paths + innactive_anno_paths + hormonal_anno_paths

    df["label"] = ['malignant'] * len(adenocarcinoma_slide_paths) + ['malignant'] * len(carcinosarcoma_slide_paths) + ['malignant'] * len(sarcoma_slide_paths) + ['malignant'] * len(hyperplasia_slide_paths) + ['malignant'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['other_benign'] * len(proliferative_slide_paths) + ['other_benign'] * len(secretory_slide_paths) +   ['othe_-benign'] * len(menstrual_slide_paths) +  ['other_benign'] * len(innactive_slide_paths) + ['other_benign'] * len(hormonal_slide_paths)



    df["tags"] = ['denocarcinoma'] * len(adenocarcinoma_slide_paths) + ['carcinosarcoma'] * len(carcinosarcoma_slide_paths) + ['sarcoma'] * len(sarcoma_slide_paths) + ['hyperplasia'] * len(hyperplasia_slide_paths) + ['other'] *  len(other_slide_paths) + ['insufficient'] * len(insufficient_slide_paths) + ['proliferative'] * len(proliferative_slide_paths) + ['secretory'] * len(secretory_slide_paths) +   ['menstrual'] * len(menstrual_slide_paths) +  ['innactive'] * len(innactive_slide_paths) + ['hormonal'] * len(hormonal_slide_paths)


    return Endometrial(root, df)




