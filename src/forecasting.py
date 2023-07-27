import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

Model = LinearRegression


class Forecasting:
    def __init__(self, data: pd.DataFrame, model_name: str):
        self.model_cpu = self.__get_model(model_name)
        self.model_mem = self.__get_model(model_name)
        self.model_disk = self.__get_model(model_name)

        self.train_data = data.iloc[: int(len(data) * 0.8)]
        self.test_data = data.iloc[int(len(data) * 0.8) :]

    @staticmethod
    def __get_model(model_name: str) -> Model:
        match model_name:
            case "linear_regression":
                return LinearRegression()
            case _:
                raise ValueError("Model not found")

    def train(self):
        x_train = pd.to_numeric(self.train_data["Timestamp"]).values.reshape(-1, 1)
        y_train_cpu = self.train_data["CPU"].values
        y_train_mem = self.train_data["Mem"].values
        y_train_disk = self.train_data["Disk"].values

        self.model_cpu.fit(x_train, y_train_cpu)
        self.model_mem.fit(x_train, y_train_mem)
        self.model_disk.fit(x_train, y_train_disk)

    def predict(self) -> tuple[np.array, np.array, np.array]:
        x_test = pd.to_numeric(self.test_data["Timestamp"]).values.reshape(-1, 1)

        cpu = self.model_cpu.predict(x_test)
        mem = self.model_mem.predict(x_test)
        disk = self.model_disk.predict(x_test)

        return cpu, mem, disk
