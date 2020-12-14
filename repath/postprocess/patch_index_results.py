from collections import namedtuple
from functools import reduce
import json
from pathlib import Path
from repath.utils.paths import project_root
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from repath.data.datasets import Dataset
from repath.data.slides import Region
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.patching.patch_index import PatchSet, PatchIndex
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.utils.geometry import Shape, Size, Point


class PatchSetResults(PatchSet):
    def __init__(self, patchset, probabilities) -> None:
        self.dataset = patchset.dataset
        self.slide_path = patchset.slide_path
        self.level = patchset.level
        self.size = patchset.size
        self.df = pd.concat((patchset.df, probabilities), axis=1)
        self.labels = patchset.labels
        self.slide_label = patchset.slide_label
        self.tags = patchset.tags

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        return self.patches_df.iterrows()[idx]

    @property
    def abs_slide_path(self):
        return self.dataset.to_abs_path(self.slide_path)

    def to_heatmap(self, class_name: str) -> np.array:
        self.df.columns = [colname.lower() for colname in self.df.columns]
        class_name = class_name.lower()

        max_rows = int(np.max(self.df.row)) + 1
        max_cols = int(np.max(self.df.column)) + 1

        # create a blank thumbnail
        thumbnail_out = np.zeros((max_rows, max_cols))

        # for each row in dataframe set the value of the pixel specified by row and column to the probability in clazz
        for rw in range(self.df.shape[0]):
            df_row = self.df.iloc[rw]
            thumbnail_out[int(df_row.row), int(df_row.column)] = df_row[class_name]

        return thumbnail_out

    def to_dict(patchset, suffix:str):
        # get the data
        info = patchset.__dict__.copy()
        info.pop('patches_df')
        info['out_path'] = patchset.slide_path.with_suffix(suffix)
        info.pop('dataset')
        return info

    def save_csv(self, output_dir):
        info = self.to_dict(self, '.csv')

        # save out the patches csv file for this slide
        csv_path = output_dir / info['out_path']
        csv_path.parents[0].mkdir(parents=True, exist_ok=True)
        self.df.to_csv(csv_path, index=False)

    def save_heatmap(self, class_name:str, output_dir: Path):
        # get the heatmap filename for this slide
        info = self.to_dict(self, '.png')
        img_path = output_dir / info['out_path']
        img_path.parents[0].mkdir(parents=True, exist_ok=True)

        # create heatmap and write out
        heatmap = self.to_heatmap(self, class_name)
        heatmap_out = np.array(np.multiply(heatmap, 255), dtype=np.uint8)
        cv2.imwrite(img_path, heatmap_out)


class PatchIndexResults(PatchIndex):
    def __init__(self, patch_index: PatchIndex) -> None:
        self.dataset = patch_index.dataset
        self.patches = patch_index.patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def save(self, output_dir: Path, results_dir: Path, heatmap_dir: Path) -> None:
        def to_dict_and_frame(patchset):
            # get the data
            info = patchset.__dict__.copy()
            info.pop('patches_df')
            info['csv_path'] = results_dir / patchset.slide_path.with_suffix('.csv')
            info['png_path'] = heatmap_dir / patchset.slide_path.with_suffix('.png')
            info.pop('dataset')
            return info, patchset.patches_df

        index_df = pd.DataFrame(columns=['slide_path', 'csv_path', 'png_path', 'slide_label', 'tags', 'level', 'patch_size', 'labels'])
        for patchset in self.patches:
            info, patches = to_dict_and_frame(patchset)

            # append the info dict to the index dataframe
            index_df = index_df.append(info, ignore_index=True)

        # tidy up a bit and save the csv
        index_df = index_df.astype({"level": int, "patch_size": int})
        output_dir.mkdir(parents=True, exist_ok=True)  # TODO: What happends if it exists already?
        index_df.to_csv(output_dir / 'results_index.csv', index=False)

    @classmethod
    def load(cls, dataset: Dataset, input_dir: Path) -> 'PatchIndex':
        def patchset_from_row(r: namedtuple) -> PatchSet:
            print(r)

            # parse all the fields
            slide_path = Path(r.slide_path)
            patch_size = int(r.patch_size)
            level = int(r.level)
            patches_df = pd.read_csv(input_dir / r.csv_path)
            print(r.tags)
            tags = str(r.tags).split(',')
            labels = json.loads(r.labels.replace('\', '\"'))

            # call the constructor
            patchset = PatchSet(dataset, slide_path, patch_size, level, 
                                patches_df, labels, r.slide_label, tags)
            return patchset

        index = pd.read_csv(input_dir / 'results_index.csv')
        patches = [patchset_from_row(r) for r in index.itertuples()]
        rtn = cls(dataset, patches)
        return rtn

    @classmethod
    def for_dataset(cls, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder) -> 'PatchIndex':
        patchsets = [PatchSet.for_slide(s, a, label, tags, dataset, tissue_detector, patch_finder) for s, a, label, tags in dataset]
        return cls(dataset, patchsets)

