from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from repath.data.annotations.annotation import AnnotationSet
from repath.data.annotations.asapxml import load_annotations
from repath.data.datasets.dataset import AnnotationSetLoader, DatasetInfoABC
from repath.data.slides.openslide import Slide
from repath.data.slides.slide_abc import SlideABC


class Camelyon16(DatasetInfoABC):
    @property
    def paths(self, train_or_test: str) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def loaders(self) -> Tuple[AnnotationSetLoader, SlideABC]:
        labels_order = ["background", "tumor", "normal"]

        def loader(file: Path) -> AnnotationSet:
            annotations = load_annotations(file)
            return AnnotationSet(annotations, self.labels, labels_order, "normal")

        return loader, Slide

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}


def training():
    return Camelyon16("train")


def testing():
    return Camelyon16("test")

