from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from repath.data.annotations import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides import SlideBase
from repath.utils.paths import project_root


class Camelyon16(Dataset):
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


def training():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # all paths are relative to the dataset 'root'
    annotation_paths = sorted([p.relative_to(root) for p in annotations_dir.glob("*.xml")])
    tumor_slide_paths = sorted([p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")])
    normal_slide_paths = sorted([p.relative_to(root) for p in normal_slide_dir.glob("*.tif")])

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ['tumor'] * len(tumor_slide_paths) + ['normal'] * len(normal_slide_paths)
    df["tags"] = ""

    return Camelyon16(root, df)


def training_small():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # all paths are relative to the dataset 'root'
    annotation_paths = sorted([p.relative_to(root) for p in annotations_dir.glob("*.xml")])
    tumor_slide_paths = sorted([p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")])
    normal_slide_paths = sorted([p.relative_to(root) for p in normal_slide_dir.glob("*.tif")])

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ['tumor'] * len(tumor_slide_paths) + ['normal'] * len(normal_slide_paths)
    df["tags"] = ""

    df = df.sample(3)

    return Camelyon16(root, df)



def testing():
    # TODO: Add this
    pass
