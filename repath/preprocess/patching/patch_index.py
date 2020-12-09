from functools import reduce
from pathlib import Path
from repath.utils.paths import project_root
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from repath.data.datasets import Dataset
from repath.data.slides import Region
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.utils.geometry import Shape, Size, Point


class PatchSet(Sequence):
    def __init__(
        self,
        dataset: Dataset,
        slide_path: Path,
        patch_size: int,
        level: int,
        patches_df: pd.DataFrame,
        labels: Dict[str, int],
        slide_label: str,
        tags: List[str]
    ) -> None:
        self.dataset = dataset
        self.slide_path = slide_path
        self.level = level
        self.patch_size = patch_size
        self.patches_df = patches_df
        self.labels = labels
        self.slide_label = slide_label
        self.tags = tags

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        return self.patches_df.items()[idx]

    def summary(self) -> pd.DataFrame:
        by_label = self.patches_df.groupby("label").size()
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
        for p in self.patches_df.itertuples(index=False):
            start = (int(p.x / factor), int(p.y / factor))
            end = (int(p.x / factor + size), int(p.y / factor + size))
            cv2.rectangle(img, start, end, p.label, -1)
        return img.astype("int")

    def save_patches(self, output_dir: Path) -> None:
        with self.dataset.slide_cls(self.slide_path) as slide:
            for row in self.patches_df.itertuples():
                region = Region.patch(row.x, row.y, self.patch_size, self.level)
                image = slide.read_region(region)
                escaped_image_path = str(self.slide_path)[:-4].replace('/', '-')
                image_path = output_dir / row.label / escaped_image_path
                image.save(image_path, '.png')

    @classmethod
    def for_slide(cls, slide_path: Path, annotation_path: Path, slide_label: str, tags: List[str], dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder):
        with dataset.slide_cls(slide_path) as slide:
            print(f"indexing {slide_path.name}")  # TODO: Add proper logging!
            annotations = dataset.load_annotations(annotation_path)
            labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
            scale_factor = 2 ** patch_finder.labels_level
            labels_image = annotations.render(labels_shape, scale_factor)
            tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
            labels_image[~tissue_mask] = 0
            df, level, size = patch_finder(labels_image)
            patchset = cls(dataset, slide_path, size, level, df, dataset.labels, slide_label, tags)
            return patchset


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

    def save(self, output_dir: Path) -> None:
        def to_dict_and_frame(patchset):
            # get the data
            info = patchset.__dict__.copy()
            info.pop('patches_df')
            info['csv_path'] = patchset.slide_path.with_suffix('.csv')
            return info, patchset.patches_df

        index_df = pd.DataFrame(columns=['slide_path', 'csv_path', 'slide_label', 'tags', 'level', 'patch_size', 'dataset_root', 'labels'])
        for patchset in self.patches:
            info, patches = to_dict_and_frame(patchset)

            # save out the patches csv file for this slide
            csv_path = output_dir / info['csv_path']
            csv_path.parents[0].mkdir(parents=True, exist_ok=True)
            patches.to_csv(csv_path, index=False)

            # append the info dict to the index dataframe
            index_df = index_df.append(info, ignore_index=True)

        # tidy up a bit and save the csv
        index_df = index_df.astype({"level": int, "patch_size": int})
        output_dir.mkdir(parents=True, exist_ok=False)
        index_df.to_csv(output_dir / 'index.csv', index=False)

    @classmethod
    def load(cls, dataset: Dataset, input_dir: Path) -> 'PatchIndex':
        def patchset_from_dict(info: Dict[str, Any]) -> PatchSet:
            csv_path = input_dir / info.pop('csv_path')
            patches_df = pd.read_csv(csv_path)
            info['patches_df'] = patches_df
            patchset = PatchSet(**info)
            return patchset

        index = pd.read_csv(input_dir / 'index.csv')
        patches = [patchset_from_dict(r.to_dict()) for r in index.iterrows()]
        rtn = cls(dataset, patches)
        return rtn

    def save_patches(self, output_dir: Path) -> None:
        for patchset in self.patches:
            patchset.save_patches(output_dir)

    @classmethod
    def for_dataset(cls, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder) -> 'PatchIndex':
        patchsets = [PatchSet.for_slide(s, a, label, tags, dataset, tissue_detector, patch_finder) for s, a, label, tags in dataset]
        return cls(dataset, patchsets)
