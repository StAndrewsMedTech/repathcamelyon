from collections import namedtuple
from itertools import chain
from pathlib import Path
import threading
from typing import List, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
import torch
from torch import nn
from torchvision import transforms

from repath.data.datasets import Dataset
from repath.preprocess.patching.patch_finder import PatchFinder
from repath.preprocess.tissue_detection.tissue_detector import TissueDetector
from repath.data.slides import Region
from repath.utils.convert import remove_item_from_dict
from repath.postprocess.prediction import inference_on_slide


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
        return self.patches_df.iloc[idx,]

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

    def save_patches(self, output_dir: Path, transforms: List[transforms.Compose] = None) -> None:
        for slide_idx, group in self.patches_df.groupby('slide_idx'):
            slide_path, _, _, _ = self.dataset[slide_idx]
            with self.dataset.slide_cls(slide_path) as slide:
                print(f"Writing patches for {self.dataset.to_rel_path(slide_path)}")
                for row in group.itertuples():
                    # read the patch image from the slide
                    region = Region.patch(row.x, row.y, self.patch_size, self.level)
                    image = slide.read_region(region)

                    # apply any transforms, as indexed in the 'transform' column
                    if transforms:
                        image = transforms[row.transform](image)

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


class CombinedIndex(object):
    def __init__(self, cps: List[CombinedPatchSet]) -> None:
        self.datasets = [cp.dataset for cp in cps]
        self.patchsizes = [cp.patch_size for cp in cps]
        self.levels = [cp.level for cp in cps]
        patches_dfs = [cp.patches_df for cp in cps]
        patches_df = pd.concat(patches_dfs, axis=0)
        cps_index = [[idx] * len(cp) for idx, cp in enumerate(cps)]
        cps_index = [item for sublist in cps_index for item in sublist]
        patches_df['cps_idx'] = cps_index
        self.patches_df = patches_df

    def __len__(self):
        return len(self.patches_df)

    @classmethod
    def for_slide_indexes(cls, indexes: List['SlidesIndex']) -> 'CombinedIndex':
        cps = [index.as_combined() for index in indexes]
        ci = cls(cps)
        return ci

    def save_patches(self, output_dir: Path, transforms: List[transforms.Compose] = None) -> None:
        for cps_idx, cps_group in self.patches_df.groupby('cps_idx'):
            for slide_idx, sl_group in cps_group.groupby('slide_idx'):
                slide_path, _, _, _ = self.datasets[cps_idx][slide_idx]
                with self.datasets[cps_idx].slide_cls(slide_path) as slide:
                    print(f"Writing patches for {self.datasets[cps_idx].to_rel_path(slide_path)}")
                    for row in sl_group.itertuples():
                        # read the patch image from the slide
                        region = Region.patch(row.x, row.y, self.patchsizes[cps_idx], self.levels[cps_idx])
                        image = slide.read_region(region)

                        # apply any transforms, as indexed in the 'transform' column
                        if transforms:
                            image = transforms[row.transform](image)

                        # get the patch label as a string
                        labels = {v: k for k, v in self.datasets[cps_idx].labels.items()}
                        label = labels[row.label]

                        # ensure the output directory exists
                        output_subdir = output_dir / label
                        output_subdir.mkdir(parents=True, exist_ok=True)

                        # write out the slide
                        rel_slide_path = self.datasets[cps_idx].to_rel_path(slide_path)
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
        abs_slide_path, self.annotation_path, self.label, tags = dataset[slide_idx]
        self.slide_path = dataset.to_rel_path(abs_slide_path)
        self.tags = [tg.strip() for tg in tags.split(';')]

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
            df, level, size = patch_finder(labels_image, slide.dimensions[patch_finder.patch_level])
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
            csv_path = output_dir / csv_path
            csv_path.parents[0].mkdir(parents=True, exist_ok=True)
            print(f"Saving {csv_path}")
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
    def load(cls, dataset: Dataset, input_dir: Path) -> 'SlidesIndex':
        def patchset_from_row(r) -> PatchSet:
            patches_df = pd.read_csv(input_dir / r.csv_path)
            return SlidePatchSet(int(r.slide_idx), dataset, int(r.patch_size), 
                                 int(r.level), patches_df)

        index = pd.read_csv(input_dir / 'index.csv')
        patches = [patchset_from_row(r) for r in index.itertuples()]
        rtn = cls(dataset, patches)
        return rtn


class SlidePatchSetResults(SlidePatchSet):
    def __init__(self, slide_idx: int, dataset: Dataset, patch_size: int, level: int, patches_df: pd.DataFrame) -> None:
        super().__init__(slide_idx, dataset, patch_size, level, patches_df)
        abs_slide_path, self.annotation_path, self.label, self.tags = dataset[slide_idx]
        self.slide_path = dataset.to_rel_path(abs_slide_path)

    @classmethod
    def predict_slide(cls, sps: SlidePatchSet, classifier: nn.Module, batch_size: int, nworkers: int,
                      transform):

        just_patch_classes = remove_item_from_dict(sps.dataset.labels, "background")
        num_classes = len(just_patch_classes)
        probs_out = inference_on_slide(sps, classifier, num_classes, batch_size, nworkers, transform)
        probs_df = pd.DataFrame(probs_out, columns=list(just_patch_classes.keys()))
        probs_df = pd.concat((sps.patches_df, probs_df), axis=1)
        patchsetresults = cls(sps.slide_idx, sps.dataset, sps.patch_size, sps.level, probs_df)
        return patchsetresults

    def to_heatmap(self, class_name: str) -> np.array:
        self.patches_df.columns = [colname.lower() for colname in self.patches_df.columns]
        class_name = class_name.lower()

        self.patches_df['column'] = np.divide(self.patches_df.x, self.patch_size)
        self.patches_df['row'] = np.divide(self.patches_df.y, self.patch_size)

        max_rows = int(np.max(self.patches_df.row)) + 1
        max_cols = int(np.max(self.patches_df.column)) + 1

        # create a blank thumbnail
        thumbnail_out = np.zeros((max_rows, max_cols))

        # for each row in dataframe set the value of the pixel specified by row and column to the probability in clazz
        for rw in range(len(self)):
            df_row = self.patches_df.iloc[rw]
            thumbnail_out[int(df_row.row), int(df_row.column)] = df_row[class_name]

        return thumbnail_out

    def save_csv(self, output_dir):
        # save out the patches csv file for this slide
        csv_path = output_dir / self.slide_path.with_suffix('.csv')
        self.patches_df.to_csv(csv_path, index=False)

    def save_heatmap(self, class_name: str, output_dir: Path):
        # get the heatmap filename for this slide
        img_path = output_dir / self.slide_path.with_suffix('.png')
        # create heatmap and write out
        heatmap = self.to_heatmap(class_name)
        heatmap_out = np.array(np.multiply(heatmap, 255), dtype=np.uint8)
        cv2.imwrite(str(img_path), heatmap_out)

    @classmethod
    def predict_slide_threaded(cls, sps: SlidePatchSet, classifier: nn.Module, batch_size: int, nworkers: int,
                      transform, device: int):

        just_patch_classes = remove_item_from_dict(sps.dataset.labels, "background")
        num_classes = len(just_patch_classes)
        probs_out = inference_on_slide_threaded(sps, classifier, num_classes, batch_size, nworkers, transform, device)
        probs_df = pd.DataFrame(probs_out, columns=list(just_patch_classes.keys()))
        probs_df = pd.concat((sps.patches_df, probs_df), axis=1)
        patchsetresults = cls(sps.slide_idx, sps.dataset, sps.patch_size, sps.level, probs_df)
        return patchsetresults


class SlidesIndexResults(SlidesIndex):
    def __init__(self, dataset: Dataset, patches: List[SlidePatchSet],
                 output_dir: Path, results_dir_name: str, heatmap_dir_name: str) -> None:
        super().__init__(dataset, patches)
        self.output_dir = output_dir
        self.results_dir_name = results_dir_name
        self.heatmap_dir_name = heatmap_dir_name

    @classmethod
    def predict_dataset(cls,
                        si: SlidesIndex,
                        classifier: nn.Module,
                        batch_size,
                        num_workers,
                        transform,
                        output_dir: Path,
                        results_dir_name: str,
                        heatmap_dir_name: str) -> 'SlidesIndexResults':

        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir / results_dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = output_dir / heatmap_dir_name
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        spsresults = []
        for sps in si:
            spsresult = SlidePatchSetResults.predict_slide(sps, classifier, batch_size, num_workers, transform)
            print(f"Saving {sps.slide_path}")
            results_slide_dir = results_dir / sps.slide_path.parents[0]
            results_slide_dir.mkdir(parents=True, exist_ok=True)
            spsresults.append(spsresult)
            spsresult.save_csv(results_dir)
            heatmap_slide_dir = heatmap_dir/ sps.slide_path.parents[0]
            heatmap_slide_dir.mkdir(parents=True, exist_ok=True)
            ### HACK since this is only binary at the moment it will always be the tumor heatmap we want need to change to work for multiple classes
            spsresult.save_heatmap('tumor', heatmap_dir)

        return cls(si.dataset, spsresults, output_dir, results_dir_name, heatmap_dir_name)

    def save_results_index(self):
        columns = ['slide_idx', 'csv_path', 'png_path', 'level', 'patch_size']
        index_df = pd.DataFrame(columns=columns)
        for ps in self.patches:
            # save out the csv file for this slide
            csv_path = self.output_dir / self.results_dir_name / ps.slide_path.with_suffix('.csv')
            png_path = self.output_dir / self.heatmap_dir_name / ps.slide_path.with_suffix('.png')

            # append information about slide to index
            info = np.array([ps.slide_idx, csv_path, png_path, ps.level, ps.patch_size])
            info = np.reshape(info, (1, 5))
            row = pd.DataFrame(info, columns=columns)
            index_df = index_df.append(row, ignore_index=True)

        # tidy up a bit and save the csv
        index_df = index_df.astype({"level": int, "patch_size": int})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(self.output_dir / 'results_index.csv', index=False)

    @classmethod
    def load_results_index(cls, dataset, input_dir, results_dir_name, heatmap_dir_name):
        def patchset_from_row(r: namedtuple) -> SlidePatchSet:
            patches_df = pd.read_csv(input_dir / r.csv_path)
            return SlidePatchSetResults(int(r.slide_idx), dataset, int(r.patch_size),
                                 int(r.level), patches_df)

        index = pd.read_csv(input_dir / 'results_index.csv')
        patches = [patchset_from_row(r) for r in index.itertuples()]
        rtn = cls(dataset, patches, input_dir, results_dir_name, heatmap_dir_name)
        return rtn

    """
    @classmethod
    def predict_dataset_threaded(cls,
                        si: SlidesIndex,
                        classifier: nn.Module,
                        batch_size,
                        num_workers,
                        transform,
                        output_dir: Path,
                        results_dir_name: str,
                        heatmap_dir_name: str) -> 'SlidesIndexResults':

        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir / results_dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = output_dir / heatmap_dir_name
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        ### experiment to distribute slides across multigpus for inference
        # find how many gpus
        ngpus = torch.cuda.device_count()

        # work out how many slides and numbers in each split
        nslides = len(si)
        splits = np.linspace(0, nslides, num=(ngpus+1))
        start_indexes = splits[0:ngpus]
        end_indexes = splits[1:]

        # shuffle slide index
        si = shuffle(si)
        si_per_gpu = []
        for ii in ngpus:
            si_gpu = si[start_indexes[ii]:end_indexes[ii]]
            si_per_gpu.append(si_gpu)

        def worker(num):
            si_thread = si_per_gpu[num]
            spsresults_thread = []
            for sps in si_thread:
                spsresult = SlidePatchSetResults.predict_slide_threaded(sps, classifier, batch_size, num_workers, transform, num)
                print(f"Saving {sps.slide_path}")
                results_slide_dir = results_dir / sps.slide_path.parents[0]
                results_slide_dir.mkdir(parents=True, exist_ok=True)
                spsresults.append(spsresult)
                spsresult.save_csv(results_dir)
                heatmap_slide_dir = heatmap_dir/ sps.slide_path.parents[0]
                heatmap_slide_dir.mkdir(parents=True, exist_ok=True)
                ### HACK since this is only binary at the moment it will always be the tumor heatmap we want need to change to work for multiple classes
                spsresult.save_heatmap('tumor', heatmap_dir)
                spsresults_thread.append(spsresult)           
            spsresults[num] = spsresults_thread
            return 

        spsresults = [0] * ngpus
        threads = []
        for i in range(ngpus):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        spsresults_flat = [item for sublist in spsresults for item in sublist]


        return cls(si.dataset, spsresults, output_dir, results_dir_name, heatmap_dir_name)
    """