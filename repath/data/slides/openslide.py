from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from openslide import OpenSlide, open_slide

from repath.data.slides.slide_abc import SlideABC, SlideImageABC, Region
from repath.utils.geometry import Point, Size


class Slide(SlideABC):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._slide = None
        self._images = []

    def open(self) -> None:
        self._slide = open_slide(str(self._path))
        image = SlideImage(self._slide)
        self._images.append(image)

    def close(self) -> None:
        self._images.clear()
        self._slide.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def images(self) -> List[SlideImageABC]:
        return self._images


class SlideImage(SlideImageABC):
    def __init__(self, openslide: OpenSlide) -> None:
        self._openslide = openslide

    def get_thumbnail(self, level: int) -> np.array:
        # TODO: check this downscaling is ok
        width, height = self._openslide.level_dimensions[level]
        im = self._openslide.get_thumbnail((width, height))
        im = im.convert("RGB")
        im = np.asarray(im)
        return im

    @property
    def dimensions(self) -> List[Size]:
        return [Size(*dim) for dim in self._openslide.level_dimensions]

    def read_region(self, region: Region) -> Image:
        return self._openslide.read_region(region.location, region.level, region.size)

    def read_regions(self, regions: List[Region]) -> Image:
        # TODO: this call could be parallelised
        regions = [self.read_region(region) for region in regions]
        return regions
