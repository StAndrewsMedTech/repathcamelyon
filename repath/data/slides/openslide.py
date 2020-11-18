from typing import List, Tuple
from pathlib import Path

import numpy as np
from openslide import open_slide, OpenSlide
from PIL import Image

from repath.data.slides.slide_abc import SlideABC, SlideImageABC
from repath.util.geometry import Point, Size


class Slide(SlideABC):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._slide = None
        self._images = []

    def open(self) -> None:
        slide = open_slide(str(self._path))
        image = SlideImage(slide)
        self._images.append(image)

    def close(self) -> None:
        self._images.clear()
        self._slide.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def images(self) -> List[SlideImage]:
        return self._images


class SlideImage(SlideImageABC):
    def __init__(self, openslide: OpenSlide) -> None:
        self._openslide = openslide

    def get_thumbnail(self, level: int) -> np.array:
        width, height = self._openslide.level_dimensions[level]
        im = self._openslide.get_thumbnail(width, height)
        im = im.convert("RGB")
        im = np.asarray(im)
        return im

    @property
    def dimensions(self, level: int) -> Size:
        return self._openslide.level_dimensions[level]

    def read_region(self, location: Point, level: int, size: Size) -> Image:
        return self._openslide.read_region(location, level, size)