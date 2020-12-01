from functools import reduce
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from repath.data.datasets.dataset import Dataset
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.preprocess.sampling import Sampler
from repath.utils.geometry import Shape


class PatchIndex:
    def __init__(
        self,
        slide_path: Path,
        patch_size: int,
        level: int,
        df: pd.DataFrame,
        labels: Dict[str, int],
    ) -> None:
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.level = level
        self.df = df
        self.labels = labels

    def summary(self) -> pd.DataFrame:
        by_label = self.df.groupby("label").count()
        return by_label

    def labels_map(self, shape: Shape, factor: int, bg: int) -> np.array:
        img = np.full(shape, bg, dtype="float")
        size = self.patch_size / factor
        for p in self.df.itertuples(index=False):
            start = (int(p.x / factor), int(p.y / factor))
            end = (int(p.x / factor + size), int(p.y / factor + size))
            cv2.rectangle(img, start, end, p.label, -1)
        return img.astype("int")


class PatchIndexSet:
    def __init__(self, dataset: Dataset, indexes: List[PatchIndex]) -> None:
        self.dataset = dataset
        self.indexes = indexes

    def summary(self) -> pd.DataFrame:
        # TODO: stack the dfs vertically and then sum down the columns!
        summaries = [s for s in self.indexes.summary()]
        # reduce(, summaries, acc)
        pass

    def save_patches(self, directory: Path) -> None:
        pass

    # constructors
    @classmethod
    def from_dataset(dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder) -> PatchIndexSet:
        def index_patches(slide_path: Path, annotation_path: Path):
            with dataset.slide_cls(slide_path) as slide:
                annotations = dataset.load_annotations(annotation_path)
                labels_shape = slide.dimensions[patch_finder.labels_level]
                scale_factor = 2 ** patch_finder.labels_level
                labels_image = annotations.render(labels_shape, scale_factor)
                tissue_mask = tissue_detector(
                    slide.get_thumbnail(patch_finder.labels_level)
                )
                labels_image[~tissue_mask] = 0
                df, level, size = patch_finder(labels_image)
                patch_index = PatchIndex(slide.path, size, level, df, dataset.labels)
                return patch_index

        indexes = [index_patches(s, a) for s, a in dataset]
        return PatchIndexSet(dataset, indexes)