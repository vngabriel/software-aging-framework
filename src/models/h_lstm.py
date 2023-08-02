import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from src.models.model import Model
from src.models.moving_average import MovingAverage
from src.utils import split_multivariate_sequences


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
    model.add(Dense(n_features))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )

    return model


class HLSTM(Model):
    def __init__(self, n_features: int, n_steps: int = 4, n_seq: int = 1):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_seq = n_seq
        self.conv_lstm_model = None
        self.train_sequence = None
        self.test_sequence = None
        self.x_train_sequence = None
        self.y_train_sequence = None
        self.x_test_sequence = None
        self.y_test_sequence = None

    @staticmethod
    def __ma_block(train_sequence: np.array, test_sequence: np.array) -> np.array:
        ma_model = MovingAverage()
        ma_model.train(
            train_sequence, test_sequence
        )  # it is trained only on train data, test data is not used
        ma_predictions = ma_model.predict(train_sequence)

        return ma_predictions

    def train(self, train_sequence: pd.DataFrame, test_sequence: pd.DataFrame):
        self.train_sequence = train_sequence
        self.test_sequence = test_sequence

        # MA block
        ma_predictions_by_resource = []
        for resource in self.train_sequence.columns:
            train_resource_sequence = self.train_sequence[resource]
            test_resource_sequence = self.test_sequence[resource]
            ma_predictions_by_resource.append(
                self.__ma_block(train_resource_sequence, test_resource_sequence)
            )

        # Data pre-processing to fit the Conv-LSTM model
        n_steps = self.n_steps

        self.x_train_sequence, self.y_train_sequence = split_multivariate_sequences(
            self.train_sequence.values, n_steps
        )
        self.x_test_sequence, self.y_test_sequence = split_multivariate_sequences(
            self.test_sequence.values, n_steps
        )

        all_ma_predictions = np.column_stack([*ma_predictions_by_resource])

        conv_lstm_x_train, conv_lstm_y_train = split_multivariate_sequences(
            all_ma_predictions, n_steps
        )

        self.x_train_sequence = self.x_train_sequence.astype(np.float32)
        self.y_train_sequence = self.y_train_sequence.astype(np.float32)
        self.x_test_sequence = self.x_test_sequence.astype(np.float32)
        self.y_test_sequence = self.y_test_sequence.astype(np.float32)
        conv_lstm_x_train = conv_lstm_x_train.astype(np.float32)
        conv_lstm_y_train = conv_lstm_y_train.astype(np.float32)

        self.n_seq = 2
        n_steps = 2

        self.x_train_sequence = self.x_train_sequence.reshape(
            (
                self.x_train_sequence.shape[0],
                self.n_seq,
                1,
                n_steps,
                self.n_features,
            )
        )
        self.x_test_sequence = self.x_test_sequence.reshape(
            (
                self.x_test_sequence.shape[0],
                self.n_seq,
                1,
                n_steps,
                self.n_features,
            )
        )
        conv_lstm_x_train = conv_lstm_x_train.reshape(
            (
                conv_lstm_x_train.shape[0],
                self.n_seq,
                1,
                n_steps,
                self.n_features,
            )
        )

        # Conv-LSTM block
        self.conv_lstm_model = create_conv_lstm(
            n_steps,
            self.n_features,
            self.n_seq,
            1e-3,
            "mse",
            ["mape", "mse", "mae"],
        )
        self.conv_lstm_model.fit(
            conv_lstm_x_train,
            conv_lstm_y_train,
            validation_data=(self.x_test_sequence, self.y_test_sequence),
            epochs=100,
            verbose=1,
        )

    def predict(self, sequence: pd.DataFrame):
        return self.conv_lstm_model.predict(sequence)

    def plot_results(self):
        sequence = np.concatenate(
            [self.train_sequence.values, self.test_sequence.values], axis=0
        )
        pred_x_train = self.conv_lstm_model.predict(self.x_train_sequence)
        pred_x_test = self.conv_lstm_model.predict(self.x_test_sequence)
        x_axis_train = np.arange(self.n_steps, len(self.train_sequence))
        x_axis_test = np.arange(len(self.train_sequence), len(sequence) - self.n_steps)

        plt.figure(figsize=(10, 6))
        for idx, resource in enumerate(self.train_sequence.columns):
            plt.subplot(len(self.train_sequence.columns), 1, idx + 1)
            plt.subplots_adjust(hspace=0.8)
            plt.plot(
                sequence[self.n_steps :, idx],
                label=f"Original Set ({resource})",
                color="blue",
            )
            plt.plot(
                x_axis_train,
                pred_x_train[:, idx],
                label=f"Predicted Train Set ({resource})",
                color="red",
                linestyle="-.",
            )
            plt.plot(
                x_axis_test,
                pred_x_test[:, idx],
                label=f"Predicted Test Set ({resource})",
                color="green",
                linestyle="-.",
            )

            plt.legend(loc="upper left", fontsize="small")
            plt.xlabel("Time")
            plt.ylabel("Resource Usage")
            plt.title(f"{resource} Usage and Prediction")
