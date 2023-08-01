from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def train(self, train_sequence: pd.DataFrame, test_sequence: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, test_sequence: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot_results(self):
        pass
