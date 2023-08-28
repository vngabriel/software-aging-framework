import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from src.models.model import Model


class MovingAverage(Model):
    def __init__(self, normalization_params: dict[str, tuple[float, float]]):
        self.normalization_params = normalization_params
        self.model = None
        self.fitted_model = None
        self.train_sequence = None
        self.test_sequence = None

    def train(self, train_sequence: pd.Series, test_sequence: pd.Series):
        self.train_sequence = train_sequence.values
        self.test_sequence = test_sequence.values
        self.model = ARIMA(train_sequence, order=(0, 0, 1))
        self.fitted_model = self.model.fit()

    def predict(self, sequence: pd.Series) -> np.ndarray:
        return self.fitted_model.predict(start=0, end=len(sequence) - 1)

    def plot_results(self):
        sequence = np.concatenate([self.train_sequence, self.test_sequence], axis=0)
        pred_x_train = self.fitted_model.predict(
            start=0, end=len(self.train_sequence) - 1
        )
        pred_x_test = self.fitted_model.predict(
            start=len(self.train_sequence), end=len(sequence) - 1
        )
        x_axis_train = np.arange(0, len(self.train_sequence))
        x_axis_test = np.arange(len(self.train_sequence), len(sequence))

        plt.plot(sequence[1:], label="Original Set", color="blue")
        plt.plot(
            x_axis_train,
            pred_x_train,
            label="Predicted Train Set",
            color="red",
            linestyle="-.",
        )
        plt.plot(
            x_axis_test,
            pred_x_test,
            label="Predicted Test Set",
            color="green",
            linestyle="-.",
        )

        plt.xlabel("Time (min)")
        plt.ylabel("Memory Used")
        plt.legend()
