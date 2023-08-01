import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

from src.models.model import Model
from src.utils import split_sequence


def create_conv_lstm(n_steps, n_features, n_seq, learning_rate, loss, metrics):
    model = Sequential(name="conv_lstm")
    model.add(
        ConvLSTM2D(
            filters=64,
            kernel_size=(1, 2),
            activation="relu",
            input_shape=(n_seq, 1, n_steps, n_features),
        )
    )
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )

    return model


class HLSTM(Model):
    def __init__(self, n_steps=4, n_features=1, n_seq=1):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_seq = n_seq
        self.ma_model = None
        self.conv_lstm_model = None
        self.train_sequence = None
        self.test_sequence = None
        self.x_train_sequence = None
        self.y_train_sequence = None
        self.x_test_sequence = None
        self.y_test_sequence = None

    def train(self, train_sequence: pd.DataFrame, test_sequence: pd.DataFrame):
        self.train_sequence = train_sequence.values
        self.test_sequence = test_sequence.values
        # MA block
        self.ma_model = ARIMA(self.train_sequence, order=(0, 0, 1))
        model_fit = self.ma_model.fit()
        ma_predictions = model_fit.predict(start=0, end=len(self.train_sequence) - 1)

        # Data pre-processing to fit the Conv-LSTM model
        self.x_train_sequence, self.y_train_sequence = split_sequence(
            self.train_sequence.tolist(), self.n_steps
        )
        x_ma_predictions, y_ma_predictions = split_sequence(
            ma_predictions.tolist(), self.n_steps
        )
        self.x_test_sequence, self.y_test_sequence = split_sequence(
            self.test_sequence.tolist(), self.n_steps
        )

        self.x_train_sequence = self.x_train_sequence.astype(np.float32)
        self.y_train_sequence = self.y_train_sequence.astype(np.float32)
        x_ma_predictions = x_ma_predictions.astype(np.float32)
        y_ma_predictions = y_ma_predictions.astype(np.float32)
        self.x_test_sequence = self.x_test_sequence.astype(np.float32)
        self.y_test_sequence = self.y_test_sequence.astype(np.float32)

        self.n_seq = 2
        self.n_steps = 2

        self.x_train_sequence = self.x_train_sequence.reshape(
            (
                self.x_train_sequence.shape[0],
                self.n_seq,
                1,
                self.n_steps,
                self.n_features,
            )
        )
        x_ma_predictions = x_ma_predictions.reshape(
            (x_ma_predictions.shape[0], self.n_seq, 1, self.n_steps, self.n_features)
        )
        self.x_test_sequence = self.x_test_sequence.reshape(
            (
                self.x_test_sequence.shape[0],
                self.n_seq,
                1,
                self.n_steps,
                self.n_features,
            )
        )

        x_ma_predictions, y_ma_predictions = x_ma_predictions[1:], y_ma_predictions[1:]

        # Conv-LSTM block
        self.conv_lstm_model = create_conv_lstm(
            self.n_steps,
            self.n_features,
            self.n_seq,
            1e-3,
            "mse",
            ["mape", "mse", "mae"],
        )
        self.conv_lstm_model.fit(
            x_ma_predictions,
            y_ma_predictions,
            validation_data=(self.x_test_sequence, self.y_test_sequence),
            epochs=100,
            verbose=1,
        )

    def predict(self, sequence: pd.DataFrame):
        return self.conv_lstm_model.predict(sequence)

    def plot_results(self):
        pred_x_train = self.conv_lstm_model.predict(self.x_train_sequence)
        pred_x_test = self.conv_lstm_model.predict(self.x_test_sequence)
        dif = len(
            np.concatenate((self.y_train_sequence, self.y_test_sequence), axis=0)
        ) - len(pred_x_test)
        axis_x_test = [(i + dif) for i in range(len(pred_x_test))]

        sequence = np.concatenate([self.train_sequence, self.test_sequence], axis=0)
        plt.plot(sequence[1:], label="Original Set", color="blue")
        plt.plot(pred_x_train, label="Predicted Train Set", color="red", linestyle="-.")
        plt.plot(
            axis_x_test,
            pred_x_test,
            label="Predicted Test Set",
            color="green",
            linestyle="-.",
        )

        plt.xlabel("Time (min)")
        plt.ylabel("Memory Used")
        plt.legend()

        plt.show()
