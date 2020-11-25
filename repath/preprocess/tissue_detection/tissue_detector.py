from abc import ABCMeta, abstractmethod

import numpy as np
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


class TissueDetector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError


class TissueDetectorOTSU(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """creates a dataframe of pixels locations labelled as tissue or not

        Based on the method proposed by wang et all
        1. Convert from RGB to HSV
        2. Perform automatic thresholding using Otsu's method on the H and S channels
        3. Combine the thresholded H and S channels

        Args:
            image: A scaled down WSI image. Must be r,g,b.

        Returns:
            An ndarray of booleans with the same dimensions as the input image
            True means foreground, False means background
        """
        # convert the image into the hsv colour space
        image_hsv = rgb2hsv(image)

        # use Otsu's method to find the thresholds for hue and saturation
        thresh_h = threshold_otsu(image_hsv[:, :, 0])
        thresh_s = threshold_otsu(image_hsv[:, :, 1])

        # mask the image to get determine which pixels with hue and saturation above their thresholds
        mask_h = image_hsv[:, :, 1] > thresh_h
        mask_s = image_hsv[:, :, 1] > thresh_s

        # combine the masks with an OR so any pixel above either threshold counts as foreground
        np_mask = np.logical_or(mask_h, mask_s)
        return np_mask

