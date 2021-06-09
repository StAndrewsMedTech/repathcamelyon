from abc import ABCMeta, abstractmethod
from typing import List

import itertools
from itertools import combinations_with_replacement
import numpy as np
from skimage import filters, feature
from skimage import img_as_float32
from concurrent.futures import ThreadPoolExecutor


class PixelFeature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class EdgeFeature(PixelFeature):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return filters.sobel(image)

class TextureFeature(PixelFeature):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        H_elems = [
            np.gradient(np.gradient(image)[ax0], axis=ax1)
            for ax0, ax1 in combinations_with_replacement(range(image.ndim), 2)
        ]
        eigvals = feature.hessian_matrix_eigvals(H_elems)
        eigvals = np.dstack((np.expand_dims(eigvals[0, :, :], axis=-1), np.expand_dims(eigvals[1, :, :], axis=-1)))

        return eigvals


class PixelFeatureDetector(object):
    def __init__(self, features_list: List, sigma_min: int = 1, sigma_max: int = 16, num_sigma: int = None, num_workers: int = None) -> None:
        if num_sigma is None:
            num_sigma = int(np.log2(sigma_max) - np.log2(sigma_min) + 1)
        sigmas = np.logspace(
            np.log2(sigma_min),
            np.log2(sigma_max),
            num=num_sigma,
            base=2,
            endpoint=True,
        )
        self.sigmas = sigmas
        self.num_workers = num_workers
        self.features_list = features_list

    def per_channel(self, img):
        def singlescale_features_singlechannel(self, img, sigma):
            gaussian_filtered = filters.gaussian(img, sigma)
            results = gaussian_filtered
            for ft in self.features_list:
                res = ft(gaussian_filtered)
                results = np.dstack((results, res))

            return results

        img = np.ascontiguousarray(img_as_float32(img))

        for idx, sig in enumerate(self.sigmas):
            out_sigma = singlescale_features_singlechannel(self, img, sig)
            if idx == 0:
                out_sigmas = out_sigma
            else:
                out_sigmas = np.dstack((out_sigmas, out_sigma))

        return out_sigmas

    def __call__(self, image: np.ndarray) -> np.array:

        for dim in range(image.shape[-1]):
            res = self.per_channel(image[..., dim])
            if dim == 0:
                all_results = res
            else:
                all_results = np.dstack((all_results, res))

        return all_results





