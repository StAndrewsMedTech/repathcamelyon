from itertools import chain
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from repath.data.datasets import Dataset
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.data.slides import Region


class PatchSet(Sequence):
    def __init__(
        self,
        dataset: Dataset,
        patch_size: int,
        level: int,
        patches_df: pd.DataFrame,
    ) -> None:
        self.dataset = dataset
        self.patch_size = patch_size
        self.level = level
        self.patches_df = patches_df

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        return self.patches_df.iterrows()[idx]

    def summary(self) -> pd.DataFrame:
        by_label = self.patches_df.groupby("label").size()
        labels = {v: k for k, v in self.dataset.labels.items()}
        count_df = by_label.to_frame().T.rename(columns = labels)
        columns = list(labels.values())
        summary = pd.DataFrame(columns=columns)
        for l in labels.values():
            if l in count_df:
                summary[l] = count_df[l]
            else:
                summary[l] = 0
        summary = summary.replace(np.nan, 0)  # if there are no patches for some classes
        return summary


class CombinedPatchSet(PatchSet):
    def __init__(self, dataset: Dataset, patch_size: int, level: int, patches_df: pd.DataFrame) -> None:
        super().__init__(dataset, patch_size, level, patches_df)
        # columns of patches_df are x, y, label, slide_idx

    def save_patches(self, output_dir: Path) -> None:
        for slide_idx, group in self.patches_df.groupby('slide_idx'):
            slide_path, _, _, _ = self.dataset[slide_idx]
            with self.dataset.slide_cls(slide_path) as slide:
                print(f"Writing patches for {self.dataset.to_rel_path(slide_path)}")
                for row in group.itertuples():
                    # read the patch image from the slide
                    region = Region.patch(row.x, row.y, self.patch_size, self.level)
                    image = slide.read_region(region)

                    # get the patch label as a string
                    labels = {v: k for k, v in self.dataset.labels.items()}
                    label = labels[row.label]

                    # ensure the output directory exists
                    output_subdir = output_dir / label
                    output_subdir.mkdir(parents=True, exist_ok=True)

                    # write out the slide
                    rel_slide_path = self.dataset.to_rel_path(slide_path)
                    slide_name_str = str(rel_slide_path)[:-4].replace('/', '-')
                    patch_filename = slide_name_str + f"-{row.x}-{row.y}.png"
                    image_path = output_dir / label / patch_filename
                    image.save(image_path)


class SlidePatchSet(PatchSet):
    def __init__(
        self, 
        slide_idx: int,
        dataset: Dataset,
        patch_size: int,
        level: int,
        patches_df: pd.DataFrame
    ) -> None:
        super().__init__(dataset, patch_size, level, patches_df)
        self.slide_idx = slide_idx
        abs_slide_path, self.annotation_path, self.label, self.tags = dataset[slide_idx]
        self.slide_path = dataset.to_rel_path(abs_slide_path)

    @classmethod
    def index_slide(cls, slide_idx: int, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder):
        slide_path, annotation_path, _, _ = dataset[slide_idx]
        with dataset.slide_cls(slide_path) as slide:
            print(f"indexing {slide_path.name}")  # TODO: Add proper logging!
            annotations = dataset.load_annotations(annotation_path)
            labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
            scale_factor = 2 ** patch_finder.labels_level
            labels_image = annotations.render(labels_shape, scale_factor)
            tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
            labels_image[~tissue_mask] = 0
            df, level, size = patch_finder(labels_image)
            patchset = cls(slide_idx, dataset, size, level, df)
            return patchset

    @property
    def abs_slide_path(self):
        return self.dataset.to_abs_path(self.slide_path)

    def open_slide(self):
        return self.dataset.slide_cls(self.abs_slide_path)


class SlidesIndex(Sequence):
    def __init__(self, dataset: Dataset, patches: List[SlidePatchSet]) -> None:
        self.dataset = dataset
        self.patches = patches

    @classmethod
    def index_dataset(cls, dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder) -> 'SlidesIndex':
        patchsets = [SlidePatchSet.index_slide(idx, dataset, tissue_detector, patch_finder) for idx in range(len(dataset))]
        return cls(dataset, patchsets)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    @property
    def patch_size(self):
        return self.patches[0].patch_size

    @property
    def level(self):
        return self.patches[0].level

    def summary(self) -> pd.DataFrame:
        summaries = [p.summary() for p in self.patches]
        slide_path = [p.slide_path for p in self.patches]
        slide_label = [p.label for p in self.patches]
        rtn = pd.concat(summaries)
        rtn['slide_path'] = slide_path
        rtn['slide_label'] = slide_label
        rtn = rtn.reset_index()
        rtn = rtn.drop('index', axis=1)
        rtn = rtn[['slide_path', 'slide_label'] + list(self.dataset.labels.keys())]
        return rtn

    def as_combined(self) -> CombinedPatchSet:
        # combine all patchsets into one
        frames = [ps.patches_df for ps in self.patches]
        slide_indexes = [[ps.slide_idx]*len(ps) for ps in self.patches]
        slide_indexes = list(chain(*slide_indexes))
        patches_df =  pd.concat(frames, axis=0)
        patches_df['slide_idx'] = slide_indexes
        return CombinedPatchSet(self.dataset, self.patch_size, self.level, patches_df)

    def save(self, output_dir: Path) -> None:
        columns = ['slide_idx', 'csv_path', 'level', 'patch_size']
        index_df = pd.DataFrame(columns=columns)
        for ps in self.patches:
            # save out the csv file for this slide
            csv_path = ps.slide_path.with_suffix('.csv')
            csv_path.parents[0].mkdir(parents=True, exist_ok=True)
            ps.patches_df.to_csv(csv_path, index=False)

            # append information about slide to index
            info = np.array([ps.slide_idx, csv_path, ps.level, ps.patch_size])
            info = np.reshape(info, (1, 4))
            row = pd.DataFrame(info, columns=columns)
            index_df = index_df.append(row, ignore_index=True)

        # tidy up a bit and save the csv
        index_df = index_df.astype({"level": int, "patch_size": int})
        output_dir.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(output_dir / 'index.csv', index=False)

    @classmethod
    def load(self, dataset: Dataset, input_dir: Path) -> 'SlidesIndex':
        def patchset_from_row(r: namedtuple) -> PatchSet:
            patches_df = pd.read_csv(input_dir / r.csv_path)
            return SlidePatchSet(int(r.slide_idx), dataset, int(r.patch_size), 
                                 int(r.level), patches_df)

        index = pd.read_csv(input_dir / 'index.csv')
        patches = [patchset_from_row(r) for r in index.itertuples()]
        rtn = cls(dataset, patches)
        return rtn
