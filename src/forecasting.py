import pandas as pd

from src.models import HLSTM, MovingAverage, Model
from src.utils import split_sets, normalize


class Forecasting:
    def __init__(self, sequence: pd.DataFrame, model_name: str, resources: list[str]):
        self.sequence = sequence
        self.resources = resources
        self.sequence = self.sequence[self.resources]
        for resource in self.resources:
            self.sequence[resource], _, _ = normalize(self.sequence[resource])
        self.train_sequence, self.test_sequence = split_sets(self.sequence, 0.8)

        self.model = self.__get_model(model_name)

    def __get_model(self, model_name: str) -> Model:
        match model_name:
            case "ma":
                return MovingAverage()
            case "h_lstm":
                return HLSTM(n_features=len(self.resources))
            case _:
                raise ValueError("Model not found")

    def train(self):
        self.model.train(self.train_sequence, self.test_sequence)

    def predict(self):
        return self.model.predict(self.test_sequence)
