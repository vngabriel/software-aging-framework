import time

import numpy as np
import pandas as pd

from src.models import HLSTM, MovingAverage, Model
from src.utils import split_sets, normalize


class Forecasting:
    def __init__(self, sequence: pd.DataFrame, model_name: str, resources: list[str]):
        self.sequence = sequence
        self.resources = resources
        self.sequence = self.sequence[self.resources]
        self.normalization_params = {}
        for resource in self.resources:
            self.sequence[resource], s_min, s_max = normalize(self.sequence[resource])
            self.normalization_params[resource] = (s_min, s_max)
        self.train_sequence, self.test_sequence = split_sets(self.sequence, 0.8)

        self.model = self.__get_model(model_name)

    def __get_model(self, model_name: str) -> Model:
        match model_name:
            case "ma":
                return MovingAverage(normalization_params=self.normalization_params)
            case "h_lstm":
                return HLSTM(
                    n_features=len(self.resources),
                    normalization_params=self.normalization_params,
                )
            case _:
                raise ValueError("Model not found")

    def train(self):
        start_time = time.time()
        self.model.train(self.train_sequence, self.test_sequence)
        end_time = time.time()
        print(f"\nTraining time: {end_time - start_time} seconds\n")

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return self.model.predict(sequence)

    def plot_results(self):
        self.model.plot_results()
