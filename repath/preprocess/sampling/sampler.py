import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.utils.convert import remove_item_from_dict

from repath.preprocess.patching.patch_index import PatchIndex

def balanced_sample(index: PatchIndex, num_samples: int, floor_samples: int = 1000) -> PatchIndex:

    # get sumamries for all slides
    summaries = index.summary()
    sum_totals = summaries.sum(axis=0, numeric_only=True)
    patch_classes = remove_item_from_dict(index.labels, "background")

    n_patches = num_samples
    for pc in patch_classes.keys():
        if sum_totals[pc] < n_patches:
            n_patches = sum_totals[pc]

    # increase no of patches for any classes with transforms

    # is min patches greater than floor
    if n_patches < floor_samples:
        n_patches = floor_samples

    # combine all patchsets into one
    all_patches = pd.DataFrame(columns=index[0].patches_df.columns)
    for patchset in index:
        all_patches = pd.concat((all_patches, patchset.patches_df), axis=0)

    # sample n patches
    sampled_patches = pd.DataFrame(columns=index[0].patches_df.columns)
    for pc in patch_classes.keys():
        class_df = sampled_patches[sampled_patches.label == pc]
        class_sample = class_df.sample(n=n_patches, axis=0, replace=False)
        sampled_patches = pd.concat((sampled_patches, class_sample), axis=0)

    return sampled_patches

