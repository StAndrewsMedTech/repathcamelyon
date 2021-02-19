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
    root = project_root() / "data" / "cervical" / "raw" 
    annotations_dir = root / "annotations"
    
    train_slide_dir = root / "train"

    #train  annotations
    train_annotation_dir = annotations_dir / "train"
    
    # all paths are relative to the dataset 'root'
    train_annotation_paths = sorted([p.relative_to(root) for p in train_annotations_dir.glob("*.xml")])
    train_slide_paths = sorted([p.relative_to(root) for p in train_slide_dir.glob("*.isyntax")])

    # load cervical data info
    cervical_data_info = pd.read_csv(root / 'iCAIRD_Cervical_Data.csv')
    
    #train slides info
    train_slides_info =  cervical_data_info.loc[cervical_data_info['train/test/valid'] == 'train']
  
    #get slide level labels
    train_slides_labels_df = train_slides_info['Category']
    
    #convert labels dataframe to a list
    train_slide_level_labels = train_slides_labels_df.values.tolist()
    
    
    #tags shows the sub-category labels
    train_tags_df = train_slides_info['subCategory']
    train_tags = train_tags_df.values.tolist()
    train_tags = ', '.join(train_tags)
   
    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = train_slide_paths 
    df["annotation"] =  train_annotation_paths 
    df["label"] = train_slide_level_labels
    df["tags"] = train_tags 

    return Cervical(root, df)



def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Test data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "cervical" / "raw" / "test"
    annotations_dir = root / "annotations"
    test_slide_dir = root / "test"
    
    test_annotation_dir = annotations_dir / "test"

    # all paths are relative to the dataset 'root'
    test_slide_paths = sorted([p.relative_to(root) for p in test_slide_dir.glob("*.isyntax")])
    test_annotation_paths = sorted([p.relative_to(root) for p in test_annotations_dir.glob("*.xml")])

    # load cervical data info
    cervical_data_info = pd.read_csv(root / 'iCAIRD_Cervical_Data.csv')
    
    #test slides info
    train_slides_info =  cervical_data_info.loc[cervical_data_info['train/test/valid'] == 'test']
   
    #get slide level labels
    test_slides_labels_df = test_slides_info['Category']
   
    #convert labels dataframe to a list
    test_slide_level_labels = test_slides_labels_df.values.tolist()
    
    #tags shows the sub-category labels
    test_tags_df = test_slides_info['subCategory']
    test_tags = test_tags_df.values.tolist()
    test_tags = ', '.join(test_tags)
   
    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = test_slide_paths 
    df["annotation"] = test_annotations_paths
    df["label"] = test_slide_level_labels
    df["tags"] = test_tags

    return Cervical(root, df)




