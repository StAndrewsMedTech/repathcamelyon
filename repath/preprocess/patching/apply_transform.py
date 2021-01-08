from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

class ApplyTransforms(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, df_in:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class SingleTransform(ApplyTransforms):
    def __init__(
        self
    ) -> None:
        pass


    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in
        df['transform'] = 1
        return df


class LiuTransform(ApplyTransforms):
    def __init__(
        self, 
        label: int,
        num_transforms: int
    ) -> None:
        self.label = label
        self.num_transforms = num_transforms


    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in
        df['transform'] = 1
        df_normal = df[df['label']!= self.label]
        df_tumor = df[df['label'] == self.label]
        tumor_rows = df_tumor.shape[0]
        df_tumor = df_tumor.loc[df_tumor.index.repeat(self.num_transforms)].reset_index(drop=True)
        df_tumor['transform'] = np.tile(list(range(1,self.num_transforms+1)), tumor_rows)
        df = pd.concat([df_normal, df_tumor], axis = 0)
        return df