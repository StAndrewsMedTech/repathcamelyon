
from abc import ABCMeta, abstractmethod
from typing import Tuple

from repath.preprocess.patching import PatchIndex


class Splitter(metaclass=ABCMeta):
    """A Splitter takes a PatchIndexSet and splits it into two subsets.

    Args:
        metaclass ([type], optional): [description]. Defaults to ABCMeta.

    Raises:
        NotImplementedError: [description]
    """
    @abstractmethod
    def __call__(self, slides: PatchIndex) -> Tuple[PatchIndex, PatchIndex]:
        raise NotImplementedError

class SlideSplitter(Splitter):
    """Splits the patches index set into two subset at a slide level, so that
    a slide (and all it's patches) are either in one subset or the other, not
    both! Used ot create train and validate sets that are sperated at a slide
    level.

    Args:
        Splitter ([type]): Inherits from the Splitter ABC.
    """
    def __init__(self, train_size: float) -> None:
        """Constructor

        Args:
            train_size (float): The ratio of slides (less than one)
        """
        assert train_size <= 1.0, "train_size must be less than or equal to one"

    def __call__(self, slides: PatchIndex) -> Tuple[PatchIndex, PatchIndex]:
        pass