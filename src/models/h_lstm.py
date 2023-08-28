import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from src.models.model import Model
from src.models.moving_average import MovingAverage
from src.utils import split_multivariate_sequences, denormalize


class HLSTM(Model):
    # paper: https://www.mdpi.com/2076-3417/12/13/6412
    # code: https://github.com/arnaldovitor/hlstm-aging
    def __init__(
        self,
        normalization_params: dict[str, tuple[float, float]],
        n_features: int,
        path_to_save_weights: str | None,
        n_steps: int = 4,
        n_seq: int = 1,
    ):
        self.normalization_params = normalization_params
        self.n_features = n_features
        self.path_to_save_weights = path_to_save_weights
        self.n_steps = n_steps
        self.n_seq = n_seq
        self.conv_lstm_model = None
        self.train_sequence = None
        self.test_sequence = None
        self.x_train_sequence = None
        self.y_train_sequence = None
        self.x_test_sequence = None
        self.y_test_sequence = None

    @staticmethod
    def create_conv_lstm(
        n_steps: int,
        n_features: int,
        n_seq: int,
        learning_rate: float,
        loss: str,
        metrics: list[str],
        path_to_load_weights: str | None,
    ):
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

        if path_to_load_weights is not None:
            try:
                model.load_weights(path_to_load_weights).expect_partial()
                print(f"\nLoaded model weights from {path_to_load_weights}\n")
            except Exception as e:
                print(
                    f"\nError loading model weights from {path_to_load_weights}: {e}\n"
                )

        return model

    def __ma_block(
        self, train_sequence: pd.Series, test_sequence: pd.Series
    ) -> np.ndarray:
        ma_model = MovingAverage(self.normalization_params)
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
        self.conv_lstm_model = self.create_conv_lstm(
            n_steps,
            self.n_features,
            self.n_seq,
            1e-3,
            "mse",
            ["mse", "mae"],
            None,
        )
        self.conv_lstm_model.fit(
            conv_lstm_x_train,
            conv_lstm_y_train,
            validation_data=(self.x_test_sequence, self.y_test_sequence),
            epochs=100,
            verbose=1,
        )
        if self.path_to_save_weights is not None:
            self.conv_lstm_model.save_weights(self.path_to_save_weights)
            print(f"\nModel saved at: {self.path_to_save_weights}\n")

    def predict(self, sequence: pd.DataFrame):
        return self.conv_lstm_model.predict(sequence)

    def load(self, path_to_load_weights: str):
        self.n_seq = 2
        n_steps = 2

        self.conv_lstm_model = self.create_conv_lstm(
            n_steps,
            self.n_features,
            self.n_seq,
            1e-3,
            "mse",
            ["mse", "mae"],
            path_to_load_weights,
        )

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
            s_min, s_max = self.normalization_params[resource]
            denormalized_sequence = denormalize(
                sequence[self.n_steps :, idx], s_min, s_max
            )
            denormalized_pred_x_train = denormalize(pred_x_train[:, idx], s_min, s_max)
            denormalized_pred_x_test = denormalize(pred_x_test[:, idx], s_min, s_max)

            plt.subplot(len(self.train_sequence.columns), 1, idx + 1)
            plt.subplots_adjust(hspace=0.8)
            plt.plot(
                denormalized_sequence,
                label=f"Original Set ({resource})",
                color="blue",
            )
            plt.plot(
                x_axis_train,
                denormalized_pred_x_train,
                label=f"Predicted Train Set ({resource})",
                color="red",
                linestyle="-.",
            )
            plt.plot(
                x_axis_test,
                denormalized_pred_x_test,
                label=f"Predicted Test Set ({resource})",
                color="green",
                linestyle="-.",
            )

            plt.legend(loc="upper left", fontsize="small")
            plt.xlabel("Time")
            plt.ylabel("Resource Usage")
            plt.title(f"{resource} Usage and Prediction")
