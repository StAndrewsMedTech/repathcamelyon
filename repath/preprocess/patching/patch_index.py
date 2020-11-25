from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from repath.utils.geometry import Shape


class PatchIndex:
    def __init__(
        self,
        slide_path: Path,
        patch_size: int,
        level: int,
        df: pd.DataFrame,
        labels: Dict[str, int],
    ) -> None:
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.level = level
        self.df = df
        self.labels = labels

    def summary(self) -> pd.DataFrame:
        pass

    def labels_map(self, shape: Shape, factor: int) -> np.array:
        pass
