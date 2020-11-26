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


class SlideImageABC(metaclass=ABCMeta):
    @abstractmethod
    def get_thumbnail(self, level: int) -> np.array:
        # TODO: this might be a bit useless - remove?
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


class SlideABC(metaclass=ABCMeta):
    """
    The Slide abstract base class defines the methodsthat all slide objects
    should implement in order to be compatable with the rest of the pipeline.

    Attributes:
        images (List[SlideImage]): List of the subimages on the slide.
    """

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def path(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def images(self) -> List[SlideImageABC]:
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
