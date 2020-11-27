from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from repath.data.annotations.annotation import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets.dataset import Dataset
from repath.data.slides.openslide import Slide
from repath.data.slides.slide import SlideBase
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
    root = project_root() / "data" / "camelyon16" / "training"

    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # path in the paths data frame are relative to the dataset 'root'
    annotation_paths = [p.relative_to(root) for p in annotations_dir.glob("*.xml")]
    tumor_slide_paths = [p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")]
    normal_slide_paths = [p.relative_to(root) for p in normal_slide_dir.glob("*.tif")]

    # turn them into a data frame
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths  # TODO: check this works - on the DGX!

    return Camelyon16(root, df)


def testing():
    # TODO: Add this
    pass
