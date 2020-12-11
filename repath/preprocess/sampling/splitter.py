
from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.preprocess.patching import PatchIndex

def split_camelyon16(index: PatchIndex, train_percent: float) -> Tuple[PatchIndex, PatchIndex]:
    pass