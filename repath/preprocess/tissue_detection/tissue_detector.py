from abc import ABCMeta, abstractmethod

import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing



class MorphologyTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class NullTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Does not apply a transform """
        return image

class SimpleClosingTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies binary closing transform """
        mask_out = binary_closing(image)
        return mask_out


class TissueDetector(metaclass=ABCMeta):
    def __init__(self, morph_transform: MorphologyTransform = NullTransform()) -> None:
        # assign values
        self.morph_transform = morph_transform

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

        # apply morphological transforms
        np_mask = self.morph_transform(np_mask)
        return np_mask


class TissueDetectorGreyScale(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """creates a dataframe of pixels locations labelled as tissue or not

        1. Convert PIL image to numpy array
        2. Convert from RGB to gray scale
        3. Get masks, any pixel that is less than a threshold (e.g. 0.8)

        Args:
        image: A scaled down WSI image. Must be r,g,b.

        Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground, False means background
        """
        # convert PIL image to numpy array
        image = np.asarray(image)

        # change pure black to pure white
        imager = image[:, :, 0] == 0
        imageg = image[:, :, 1] == 0
        imageb = image[:, :, 2] == 0
        image_mask = np.expand_dims(np.logical_and(np.logical_and(imager, imageg), imageb), axis=-1)
        image = np.where(image_mask, [255,255,255], image)
        image = np.array(image, dtype=np.uint8)

        # convert to gray-scale
        image_grey = rgb2gray(image)

        # get masks, any pixel that is less than 0.8
        np_mask = np.less_equal(image_grey, 0.8)

        # apply morphological transforms
        np_mask = self.morph_transform(np_mask)

        return np_mask


class TissueDetectorAll(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """creates a dataframe of all pixels in image labelled as foreground

        Args:
        image: A scaled down WSI image. Must be r,g,b.

        Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground
        """
        # convert PIL image to numpy array
        image = np.asarray(image)

        # get masks, any pixel that is less than 0.8
        np_mask = np.array(np.ones(image.shape[0:2]), dtype=bool)

        # apply morphological transforms
        np_mask = self.morph_transform(np_mask)

        return np_mask

