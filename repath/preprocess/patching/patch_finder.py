from abc import ABCMeta, abstractmethod
from typing import Dict
from random import randint

from repath.data.patching.patch_index import PatchIndex
from repath.utils.convert import to_frame_with_locations
from repath.utils.filters import pool2d

import pandas as pd
import numpy as np


class PatchFinder(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, labels_image: np.array) -> PatchIndex:
        raise NotImplementedError


class GridPatchFinder:
    def __init__(
        self,
        labels: Dict[str, int],
        labels_level: int,
        patch_level: int,
        patch_size: int,
        stride: int,  # defined in terms of the labels image space
        border: int = 0,
        jitter: int = 0,
    ) -> None:
        """ Note that the assumption is that the same settings will be used for a number of different patches.

        Args:
            labels (Dict[str, int]): Dict mapping string labels to indices in the labels image.
            labels_level (int): The magnification level of the labels image.
            patch_level (int): The magnification level at which to extract the pixels data for the patches.
            patch_size (int): The width and height of the patches in pixels at patches_level magnification.
            stride (int): The horizontal and vertical distance between each patch (the stide of the window).
            border (int, optional): [description]. Defaults to 0.
            jitter (int, optional): [description]. Defaults to None.
        """

        # assign values
        self.labels = labels
        self.labels_level = labels_level
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.stride = stride
        self.border = border
        self.jitter = jitter

        # some assumptions
        # 1. patch_size is some integer multiple of a pixel at labels_level
        # 2. patch_level is equal to or below labels_level
        # 3. stride is some integer multiple of a pixel at labels_level

    def __call__(self, labels_image: np.array) -> PatchIndex:
        """Patch finders can be called with an array of rendered annotations and produce a patch index.

        Args:
            labels_image (np.array): An array containing a label index for each pixel at some magnification level.

        Returns:
            PatchIndex: A patch index containing data about how to retieve and label the patches for the slide.
        """
        # generate the data frame
        # the data frame has the following columns:
        # the frame is basically a list of Regions but
        # they all share the same level, which is patch_level
        # x: float
        # y: float
        # label: str

        # to create this we do the following steps:
        # 1. work out how many pixels in the label image there are for each patch
        #   the patch is patch_size at patch_level
        #   the scale difference between label_level and patch_level is 2 ** (patch_level - labels_level)
        #   so a patch is patch_size / scale_difference
        scale_factor = 2 ** (self.labels_level - self.patch_level)

        # 2. apply a convolution over the label image so there is one pixel per patch (using maxpooling in order to exclude conficting labels), this enables the stide to work
        #   The kernal size of the maxpooling operation needs to scale from the size of one patch at label level to one pixel in the new image.
        kernel_size = int(self.patch_size / scale_factor)
        patch_labels = pool2d(labels_image, kernel_size, kernel_size, 0)

        # 3. use to_location from utils to change it into a data frame with a row and col
        df = to_frame_with_locations(patch_labels, "label")

        # 4. for each row, use the row and col to compute the x and y at the patch level
        df.row *= self.patch_size
        df.column *= self.patch_size
        df = df.rename(columns={"row": "y", "column": "x"})
        df = df.reindex(columns=["x", "y", "label"])

        # 5. for each row, add the border
        # self.border = border
        # TODO: Add in the boarder transform

        #   subtract the boarder from the top-right point and add it to the patch_size
        # 6. for each row, add the jitter
        if self.jitter != 0:

            def jitter(val: int) -> int:
                val = val - randint(0, self.jitter)
                return np.minimum(val, 0)

            df["x"] = df["x"].apply(jitter)
            df["y"] = df["y"].apply(jitter)

        return df


class RandomPatchFinder:
    pass
