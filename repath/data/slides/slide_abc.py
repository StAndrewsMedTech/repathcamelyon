from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from repath.util.geometry import Point, Size, Address

class Region:
    level: int
    location: Point
    size: Size


class SlideABC(metaclass=ABC):
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


class SlideImageABC(metaclass=ABC):
    @abstractmethod
    def get_thumbnail(self, level: int) -> np.array:
        raise NotImplementedError

    @abstractproperty
    def dimensions(self, level: int) -> Size:
        raise NotImplementedError

    @abstractmethod
    def read_region(self, region: Region) -> Image:
        raise NotImplementedError

    @abstractmethod
    def read_regions(self, region: List[Region]) -> Image:
        raise NotImplementedError


class PatchExtractorRegular:
    def __init__(self, image: SlideImageABC, level: int, patch_size: int, stride: int, border: int):
        pass

    @property
    def shape_in_patches(self):
        pass

    def read_patch(self, address: Address) -> np.array:
        pass

    def read_patches(self, addresses: List[Address]) -> List[np.array]:
        pass

class PatchExtractorRandom:
    pass