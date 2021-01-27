from math import ceil
from multiprocessing import Pool, set_start_method

from itertools import cycle
from collections import namedtuple
from typing import Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import pytorch_lightning as pl

from repath.data.datasets import Dataset
from repath.postprocess.slide_dataset import SlideDataset
from repath.preprocess.patching.patch_index import PatchSet, SlidesIndex, SlidePatchSet
from repath.postprocess.prediction import evaluate_on_device
from repath.utils.convert import remove_item_from_dict
from torchvision.transforms import Compose

import os


def predict_slide(args: Tuple[SlidePatchSet, int, Compose, pl.LightningModule, int]) -> 'SlidePatchSetResults':
    
    ps, device_idx, transform, model, batch_size, border, jitter, output_dir, results_dir_name, heatmap_dir_name = args
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    results_all = []
    for sps in ps:
        dataset = SlideDataset(sps, transform)
        dataset.open_slide()
        test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
        just_patch_classes = remove_item_from_dict(sps.dataset.labels, "background")
        num_classes = len(just_patch_classes)
        probs_out = evaluate_on_device(model, device, test_loader, num_classes, device_idx)
        ntransforms = 1
        npreds = int(len(dataset) * ntransforms)
        probs_out = probs_out[0:npreds, :]

        ''' - TODO: ADD IN FOR MULTIPLE TRANSFORMS
        if ntransforms > 1:
            prob_rows = probabilities.shape[0]
            prob_rows = int(prob_rows / ntransforms)
            probabilities_reshape = np.empty((prob_rows, num_classes))
            for cl in num_classes:
                class_probs = probabilities[:, cl]
                class_probs = np.reshape(class_probs, (ntransforms, prob_rows)).T
                class_probs = np.mean(class_probs, axis=1)
                probabilities_reshape[:, cl] = class_probs
            probabilities = probabilities_reshape
        '''

        probs_df = pd.DataFrame(probs_out, columns=list(just_patch_classes.keys()))
        probs_df = pd.concat((sps.patches_df, probs_df), axis=1)
        dataset.close_slide()
        results = SlidePatchSetResults(sps.slide_idx, sps.dataset, sps.patch_size, sps.level, probs_df, border, jitter)
        results.save_csv(output_dir / results_dir_name / results.slide_path.parents[0])
        results.save_heatmap(output_dir / heatmap_dir_name / results.slide_path.parents[0])
        results_all.append(results)
    return results_all
   
class SlidePatchSetResults(SlidePatchSet):
    def __init__(self, slide_idx: int, dataset: Dataset, patch_size: int, level: int, patches_df: pd.DataFrame, 
                 border: int = 0, jitter: int = 0) -> None:
        super().__init__(slide_idx, dataset, patch_size, level, patches_df)
        abs_slide_path, self.annotation_path, self.label, self.tags = dataset[slide_idx]
        self.slide_path = dataset.to_rel_path(abs_slide_path)
        self.border = border
        self.jitter = jitter

    def to_heatmap(self, class_name: str) -> np.array:
        self.patches_df.columns = [colname.lower() for colname in self.patches_df.columns]
        class_name = class_name.lower()

        # amount to add to find top left
        add_top_left = ceil(self.border / 2) + self.jitter

        # find top left positions without border and jitter
        top_x = np.add(self.patches_df.x, add_top_left)
        left_y = np.add(self.patches_df.y, add_top_left)

        # find core patch size
        base_patch_size = self.patch_size - self.border

        # remove border and convert to column, row
        self.patches_df['column'] = np.divide(top_x, base_patch_size)
        self.patches_df['row'] = np.divide(left_y, base_patch_size)

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
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / self.slide_path.with_suffix('.csv')
        self.patches_df.to_csv(csv_path, index=False)

    def save_heatmap(self, class_name: str, output_dir: Path):
        # get the heatmap filename for this slide
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / self.slide_path.with_suffix('.png')
        # create heatmap and write out
        heatmap = self.to_heatmap(class_name)
        heatmap_out = np.array(np.multiply(heatmap, 255), dtype=np.uint8)
        cv2.imwrite(str(img_path), heatmap_out)


class SlidesIndexResults(SlidesIndex):
    def __init__(self, dataset: Dataset, patches: List[SlidePatchSet],
                 output_dir: Path, results_dir_name: str, heatmap_dir_name: str) -> None:
        super().__init__(dataset, patches)
        self.output_dir = output_dir
        self.results_dir_name = results_dir_name
        self.heatmap_dir_name = heatmap_dir_name

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
    def load_results_index(cls, dataset, input_dir, results_dir_name, heatmap_dir_name, border=0, jitter=0):
        def patchset_from_row(r: namedtuple) -> SlidePatchSet:
            patches_df = pd.read_csv(input_dir / r.csv_path)
            return SlidePatchSetResults(int(r.slide_idx), dataset, int(r.patch_size),
                                 int(r.level), patches_df, border, jitter)

        index = pd.read_csv(input_dir / 'results_index.csv')
        patches = [patchset_from_row(r) for r in index.itertuples()]
        rtn = cls(dataset, patches, input_dir, results_dir_name, heatmap_dir_name)
        return rtn

    @classmethod
    def predict(cls, slide_index: SlidesIndex, model, transform, batch_size, output_dir, results_dir_name, heatmap_dir_name, 
                border=0, jitter=0) -> 'SlidesIndexResults':
        slide_index = [slide_index[1:3],slide_index[4:6],slide_index[7:9],slide_index[10:12],
                        slide_index[13:15],slide_index[16:18],slide_index[19:21],slide_index[22:24]]
        # spawn a process to predict for each slide
        ngpus = torch.cuda.device_count()
        slides = zip(slide_index, range(ngpus), [transform]*ngpus, [model]*ngpus, [batch_size] * 
                        ngpus, [border] *ngpus, [jitter] * ngpus, [output_dir] *ngpus, [results_dir_name]*ngpus, [heatmap_dir_name]*ngpus)
        pool = Pool()
        results = pool.map(predict_slide, slides)
        pool.join()
        return cls(slide_index.dataset, results, output_dir, results_dir_name, heatmap_dir_name)
