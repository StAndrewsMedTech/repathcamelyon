from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.preprocess.patching.patch_index import PatchIndex

class Sampler(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, patches: PatchIndex, num_samples: int) -> PatchIndex:
        raise NotImplementedError


class StratifiedPatchSampler(Sampler):
    """This class performs a stratified sampling over all slides in a patches 
    index in order to balance the amount of patches with different labels in the data.
    
    It samples from each slide based on the number of patches.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, patches: PatchIndex, num_samples: int):
        pass
