from abc import ABCMeta, abstractmethod
from pathlib import Path
from repath.data.slides.openslide import Slide
from repath.data.annotations.asapxml import load_annotations
from typing import Callable, List, Tuple

import pandas as pd
from repath.data.annotations.annotation import Annotation, AnnotationSet
from repath.data.slides.slide_abc import SlideABC

AnnotationSetLoader = Callable[[Path], AnnotationSet]


class DatasetInfoABC(metaclass=ABCMeta):
    @property
    @abstractmethod
    def paths(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def annotation_loader(self) -> Tuple[AnnotationSetLoader, SlideABC]:
        raise NotImplementedError

    @property
    def labels(self) -> Dict[str, int]:
        raise NotImplementedError


class Dataset:
    """Container class for all of the slides and information about a dataset
    How do we use this class?
    - Transform it by iterating over it!
    - So it's a sequence?
    """

    def __init__(self, info: DatasetInfoABC) -> None:
        # process the paths_df (has two columns 'slide', 'annotation')
        self.info = info
        self.wsis = []  # TODO: refactor to a different name. wsis is confusing.

    def load(self) -> None:
        load_annots, Slide = self.info.loaders
        # TODO: line below doesn't work for slides with multiple images
        self.wsis = [
            (Slide(row.slide), load_annots(row.annotation))
            for row in self.paths.itertuples()
        ]

    def make_patches(
        self,
        labels_level: int,
        patch_finder: PatchFinder,
        tissue_detector: TissueDetector,
    ) -> None:
        load_annots, Slide = self.info.loaders
        for wsi in self.wsis:
            with Slide(wsi.slide) as slide:
                for image in slide.images:
                    thumb = image.get_thumbnail(labels_level)
                    scale_factor = 2 ** labels_level
                    labels_image = wsi

            # 1. create the tissue mask

            # 2.

    def save_patches(self, path: Path) -> None:
        pass
