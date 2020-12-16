import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.utils.convert import remove_item_from_dict

from repath.preprocess.patching.patch_index import PatchIndex, PatchSet

def balanced_sample(index: PatchIndex, num_samples: int, floor_samples: int = 1000) -> PatchIndex:

    # get sumamries for all slides
    summaries = index.summary()
    sum_totals = summaries.sum(axis=0, numeric_only=True)
    print(sum_totals)
    patch_classes = remove_item_from_dict(index.dataset.labels, "background")

    n_patches = num_samples
    for pc in patch_classes.keys():
        if sum_totals[pc] < n_patches:
            n_patches = sum_totals[pc]

    # increase no of patches for any classes with transforms

    # is min patches greater than floor
    if n_patches < floor_samples:
        n_patches = floor_samples

    # combine all patchsets into one
    all_patches = index.to_patchset()
    level = index[0].level
    size = index[0].size

    # sample n patches
    sampled_patches = PatchSet(index.dataset, size, level, all_patches)
    for key, value in patch_classes.items():
        class_n_patches = min(sum_totals[key], n_patches)
        class_df = all_patches.patches_df[all_patches.patches_df.label == value]
        print(class_df.shape, class_n_patches)
        class_sample = class_df.sample(n=class_n_patches, axis=0, replace=False)
        sampled_patches.patches_df = pd.concat((sampled_patches.patches_df, class_sample), axis=0)

    return sampled_patches

