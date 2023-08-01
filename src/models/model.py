from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def train(
        self,
        train_sequence: pd.DataFrame | pd.Series,
        test_sequence: pd.DataFrame | pd.Series,
    ):
        pass

    @abstractmethod
    def predict(self, sequence: pd.DataFrame | pd.Series) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot_results(self):
        pass
