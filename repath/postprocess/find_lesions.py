import numpy as np
import pandas as pd
from skimage.measure import regionprops

from abc import ABC, abstractmethod


class LesionFinder(ABC):
    @abstractmethod
    def find_lesions(self, heatmap: np.array, labelled_image: np.array) -> pd.DataFrame:
        pass


class LesionFinderBasic(LesionFinder):
    def find_lesions(self, heatmap: np.array, labelled_image: np.array, pixels_per_pixel_thumb: int = 256) -> pd.DataFrame:
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # get region_props
        labelled_image = np.array(labelled_image, dtype='int')
        reg_props = regionprops(labelled_image, intensity_image=heatmap, coordinates='xy')

        # get centre for each region
        img_cents = [reg.centroid for reg in reg_props]

        # convert img_cents to level zero scale
        img_cents = np.multiply(img_cents, pixels_per_pixel_thumb)

        # probability score for each region
        prob_score = [np.sum(reg.intensity_image) for reg in reg_props]

        # save number of pixels per region
        img_area = [reg.area for reg in reg_props]

        # combine to give output
        if len(prob_score) > 0:
            output_df = pd.DataFrame(
                np.hstack((np.reshape(np.array(prob_score), (len(prob_score), 1)), np.array(img_cents),
                           np.reshape(np.array(img_area), (len(img_area), 1)))),
                columns=["prob_score", "centre_row", "centre_col", "pixels"])
        else:
            output_df = pd.DataFrame(columns=["prob_score", "centre_row", "centre_col", "pixels"])
        return output_df


class LesionFinderWang(LesionFinder):
    def find_lesions(self, heatmap: np.array, heatmap_hnm: np.array, labelled_image: np.array) -> pd.DataFrame:
        assert heatmap.dtype == 'float' and np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Heatmap in wrong format"

        # get region_props
        reg_props = regionprops(labelled_image, intensity_image=heatmap, coordinates='xy')
        reg_props_hnm = regionprops(labelled_image, intensity_image=heatmap_hnm, coordinates='xy')

        # get centre for each region
        img_cents = [reg.centroid for reg in reg_props]

        # convert img_cents to level zero scale
        img_cents = np.multiply(img_cents, 256)

        # probability score for each region
        prob_score_orig = [np.sum(reg.intensity_image) for reg in reg_props]
        prob_score_hnm = [np.sum(reg.intensity_image) for reg in reg_props_hnm]
        prob_score = np.divide(np.add(prob_score_orig, prob_score_hnm), 2)

        # save number of pixels per region
        img_area = [reg.area for reg in reg_props]

        # combine to give output
        if len(prob_score) > 0:
            output_df = pd.DataFrame(
                np.hstack((np.reshape(np.array(prob_score), (len(prob_score), 1)), np.array(img_cents),
                           np.reshape(np.array(img_area), (len(img_area), 1)))),
                columns=["prob_score", "centre_row", "centre_col", "pixels"])
        else:
            output_df = pd.DataFrame(columns=["prob_score", "centre_row", "centre_col", "pixels"])
        return output_df
