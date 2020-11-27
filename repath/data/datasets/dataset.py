from abc import ABCMeta, abstractmethod
from collections import Sequence
from pathlib import Path
from repath.data.slides.slide import SlideBase
from typing import Callable, Dict, Tuple

import pandas as pd
from repath.data.annotations.annotation import AnnotationSet


class Dataset(Sequence, metaclass=ABCMeta):
    """ A data set is an object that represents a set of slides and their annotations.

    It can be used to load and iterate over a set of slides and their annotations.
    This is an abstract base class where classes that represent specific data sets
    should overload the load_annotations and slide_cls methods. See their descriptions
    for details.
    It implements the Sequence protocol so that it can be iterated over.

    Args:
        Sequence ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, root: Path, paths: pd.DataFrame) -> None:
        # process the paths_df (has two columns 'slide', 'annotation')
        self.root = root
        self.paths = paths

    @abstractmethod
    def load_annotations(file: Path) -> AnnotationSet:
        raise NotImplementedError

    @property
    @abstractmethod
    def slide_cls(self) -> SlideBase:
        raise NotImplementedError

    @property
    def labels(self) -> Dict[str, int]:
        raise NotImplementedError

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        row = self.paths.iloc[idx]
        return row["slide"], row["annotation"]
