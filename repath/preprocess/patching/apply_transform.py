from abc import ABCMeta, abstractmethod


class ApplyTransforms(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, df_in:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class SingleTransform(ApplyTransforms):
    def __init__(
        self
    ) -> None:

    self.df_in = df_in

    def __call__(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in
        df['transform'] = 1
        return df


    