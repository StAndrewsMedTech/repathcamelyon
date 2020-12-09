from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from PIL import Image
from repath.utils.geometry import Point, Size


class Region(NamedTuple):
    level: int
    location: Point
    size: Size

    @classmethod
    def patch(cls, x, y, size, level):
        location = Point(x, y)
        size = Size(size, size)
        return Region(level, location, size)


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
        size = self.dimensions[level]
        region = Region(level=level, location=(0, 0), size=size)
        im = self.read_region(region)
        im = im.convert("RGB")
        im = np.asarray(im)
        return im
