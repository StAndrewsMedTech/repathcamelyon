import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from repath.utils.convert import remove_item_from_dict

from repath.preprocess.patching import CombinedIndex, CombinedPatchSet, SlidesIndex

def balanced_sample(indexes: List[SlidesIndex], num_samples: int, floor_samples: int = 1000) -> CombinedPatchSet:

    index = CombinedIndex.for_slide_indexes(indexes)

    # work out how many of each type of patches you have in the index
    labels = np.unique(index.patches_df.label)
    sum_totals = [np.sum(index.patches_df.labels == label) for label in labels]

    # find the count for the class that has the lowest count, so we have balanced classes
    n_patches = min(sum_totals)

    # limit the count for each class to the number of samples we want
    n_patches = min(n_patches, num_samples)

    # make sure that have a minimun number of samples for each class if available
    # classes with smaller that floor with remain the same
    n_patches = max(n_patches, floor_samples)
    sum_totals = np.min(sum_totals, n_patches)  # cap the number sample from each class to n_patches

    # sample n patches
    sampled_patches = pd.DataFrame(columns=index.patches_df.columns)
    for idx, label in enumerate(labels):
        class_df = index.patches_df[index.patches_df.label == label]
        class_sample = class_df.sample(n=sum_totals[idx], axis=0, replace=False)
        sampled_patches = pd.concat((sampled_patches, class_sample), axis=0)

    index.patches_df = sampled_patches
    return index