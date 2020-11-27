from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from repath.utils.geometry import Point, Size


class Region:
    level: int
    location: Point
    size: Size


class SlideBase(metaclass=ABCMeta):
    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    @abstractmethod
    def path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> List[Size]:
        raise NotImplementedError

    @abstractmethod
    def read_region(self, region: Region) -> Image:
        raise NotImplementedError

    @abstractmethod
    def read_regions(self, regions: List[Region]) -> Image:
        raise NotImplementedError

    def get_thumbnail(self, level: int) -> np.array:
        # TODO: check this downscaling is ok
        x, y, w, h = self.dimensions[level]
        region = Region(level=level, location=(x, y), size=(w, h))
        im = self.read_region(region)
        im = im.convert("RGB")
        im = np.asarray(im)
        return im
