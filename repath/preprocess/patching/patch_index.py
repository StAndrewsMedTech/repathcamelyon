from functools import reduce
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from repath.data.datasets.dataset import Dataset
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.utils.geometry import Shape


class PatchSet:
    def __init__(
        self,
        dataset_root: Path,
        slide_path: Path,
        patch_size: int,
        level: int,
        df: pd.DataFrame,
        labels: Dict[str, int],
        slide_label: str,
        tags: str
    ) -> None:
        self.dataset_root = dataset_root
        self.slide_path = slide_path.relative_to(dataset_root)
        self.patch_size = patch_size
        self.level = level
        self.df = df
        self.labels = labels
        self.slide_label = slide_label
        self.tags = tags

    def summary(self) -> pd.DataFrame:
        by_label = self.df.groupby("label").size()
        labels = {v: k for k, v in self.labels.items()}
        count_df = by_label.to_frame().T.rename(columns = labels)
        columns = ['slide'] + list(labels.values())
        summary = pd.DataFrame(columns=columns)
        for l in labels.values():
            if l in count_df:
                summary[l] = count_df[l]
            else:
                summary[l] = 0
        summary['slide'] = self.slide_path
        summary = summary.replace(np.nan, 0)  # if there are no patches for some classes
        return summary

    def labels_map(self, shape: Shape, factor: int, bg: int) -> np.array:
        # TODO: This is hard to use because you need to know the shape and scale
        # perhaps it should just ask for a level?
        img = np.full(shape, bg, dtype="float")
        size = self.patch_size / factor
        for p in self.df.itertuples(index=False):
            start = (int(p.x / factor), int(p.y / factor))
            end = (int(p.x / factor + size), int(p.y / factor + size))
            cv2.rectangle(img, start, end, p.label, -1)
        return img.astype("int")


class PatchIndex(Sequence):
    def __init__(self, dataset: Dataset, patches: List[PatchSet]) -> None:
        self.dataset = dataset
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def summary(self) -> pd.DataFrame:
        summaries = [s.summary() for s in self.patches]
        rtn = pd.concat(summaries)
        rtn = rtn.reset_index()
        rtn = rtn.drop('index', axis=1)
        return rtn

    def save(self, directory: Path) -> None:
        """Serialise the PatchSet to a set of csv files.

        There is one csv file for each patch set in the index.
        There is a csv with the details of each set in a row.

        Args:
            directory (Path): The directory to output the files to.
        """
        

    def export_patches(self, directory: Path) -> None:
        pass


def index_patches(dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder) -> 'PatchIndex':
    def index_patches(slide_path: Path, annotation_path: Path, slide_label: str, tags: str):
        with dataset.slide_cls(slide_path) as slide:
            print(f"indexing {slide_path.name}")  # TODO: Add proper logging!
            annotations = dataset.load_annotations(annotation_path)
            labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
            scale_factor = 2 ** patch_finder.labels_level
            labels_image = annotations.render(labels_shape, scale_factor)
            tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
            labels_image[~tissue_mask] = 0
            df, level, size = patch_finder(labels_image)
            patch_index = PatchSet(dataset.root, slide_path, size, level, df, dataset.labels, slide_label, tags)
            return patch_index

    indexes = [index_patches(s, a, label, tags) for s, a, label, tags in dataset]
    return PatchIndex(dataset, indexes)
