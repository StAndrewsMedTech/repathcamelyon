import os
import pandas as pd
from pathlib import Path
from repath.utils.paths import project_root


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
    root = project_root() / "iCAIRD"  
    annotations_dir = root / "annotations"
    
    csv_path = root / "iCAIRD_Cervical_Data.csv"

    cervical_df = pd.read_csv(csv_path)
    cervical_train = cervical_df[cervical_df['train/valid/test']=='train']

    annotations = cervical_train['Image Filename'].replace({'isyntax':'txt'}, regex=True)

    df = pd.DataFrame()
    df["slide"] =  str(os.path.relpath(root))+ '/' + cervical_train['Image Filename']
    df["label"] = cervical_train['Category']
    df["annotation"] = str(os.path.relpath(annotations_dir)) + '/' + annotations
    df["tags"] = cervical_train['subCategory']
   
    return Cervical(root, df)


    def testing():
     """ Generated a data-frame of slide_path, annotation_path, label and tags for test dataset.

    Returns:
        DataFrame (pd.DataFrame): Train data frame
    """
    # set up the paths to the slides and annotations
    root = project_root() / "iCAIRD"  
    annotations_dir = root / "annotations"
    

    csv_path = root / "iCAIRD_Cervical_Data.csv"

    cervical_df = pd.read_csv(csv_path)
    cervical_test= cervical_df[cervical_df['train/valid/test']=='test']

    annotations = cervical_test['Image Filename'].replace({'isyntax':'txt'}, regex=True)

    df = pd.DataFrame()
    df["slide"] =  str(os.path.relpath(root)) + '/' + cervical_test['Image Filename']
    df["label"] = cervical_test['Category']
    df["annotation"] = str(os.path.relpath(annotations_dir)) + '/' + annotations
    df["tags"] = cervical_test['subCategory']
   
    return Cervical(root, df)