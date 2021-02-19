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
    root = project_root() / "data" / "endometrial" / "raw" 
    annotations_dir = root / "annotations"
    
    train_slide_dir = root / "train"

    #train  annotations
    train_annotation_dir = annotations_dir / "train"
    
    # all paths are relative to the dataset 'root'
    train_annotation_paths = sorted([p.relative_to(root) for p in train_annotations_dir.glob("*.txt")])
    train_slide_paths = sorted([p.relative_to(root) for p in train_slide_dir.glob("*.isyntax")])

    # load endometrial data info
    endometrial_data_info = pd.read_csv(root / 'iCAIRD_Endometrial_Data.csv')
    
    #train slides info
    train_slides_info =  endometrial_data_info.loc[endometrial_data_info['train/test/valid'] == 'train']
  
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

    return Endometrial(root, df)

def testing():
    """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Test data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "endometrial" / "raw" / "test"
    annotations_dir = root / "annotations"
    test_slide_dir = root / "test"
    
    test_annotation_dir = annotations_dir / "test"

    # all paths are relative to the dataset 'root'
    test_slide_paths = sorted([p.relative_to(root) for p in test_slide_dir.glob("*.isyntax")])
    test_annotation_paths = sorted([p.relative_to(root) for p in test_annotations_dir.glob("*.txt")])

    # load endometrial data info
    endometrial_data_info = pd.read_csv(root / 'iCAIRD_Endometrial_Data.csv')
    
    #test slides info
    train_slides_info =  endometrial_data_info.loc[endometrial_data_info['train/test/valid'] == 'test']
   
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

    return Endometrial(root, df)




