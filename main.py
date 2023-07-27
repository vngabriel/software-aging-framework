import argparse
import sys
import time

import pandas as pd
from matplotlib import pyplot as plt

from src.forecasting import Forecasting
from src.monitor import ResourceMonitorProcess


class Framework:
    def __init__(
        self,
        monitoring_time_in_seconds: int,
        monitoring_interval_in_seconds: int,
        prediction_interval_in_seconds: int,
        filename: str,
        model: str,
        save_plot: bool,
        run_in_real_time: bool,
    ):
        self.monitoring_time_in_seconds = monitoring_time_in_seconds
        self.monitoring_interval_in_seconds = monitoring_interval_in_seconds
        self.prediction_interval_in_seconds = prediction_interval_in_seconds
        self.filename = filename
        self.model_name = model
        self.save_plot = save_plot
        self.run_in_real_time = run_in_real_time
        self.forecasting = None
        self.monitor_process = ResourceMonitorProcess(
            self.monitoring_interval_in_seconds, self.filename
        )

    def run(self):
        if self.run_in_real_time:
            self.__run_real_time()
        else:
            self.__run_experiment()

    def __run_monitoring(self):
        self.monitor_process.start()
        self.__countdown()
        self.__stop()

    def __run_experiment(self):
        self.__run_monitoring()

        dataframe = pd.read_csv(self.filename)
        dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"])

        self.forecasting = Forecasting(dataframe, self.model_name)
        self.forecasting.train()
        self.__plot_graph()

    def __run_real_time(self):
        ...

    def __stop(self):
        self.monitor_process.terminate()

    def __print_progress_bar(self, current_second, text):
        progress_bar_size = 50
        current_progress = (current_second + 1) / self.monitoring_time_in_seconds
        sys.stdout.write(
            f"\r{text}: [{'=' * int(progress_bar_size * current_progress):{progress_bar_size}s}] "
            f"{current_second + 1}/{self.monitoring_time_in_seconds} seconds"
        )
        sys.stdout.flush()

    def __countdown(self):
        for current_second in range(self.monitoring_time_in_seconds):
            self.__print_progress_bar(current_second, "Monitoring")
            time.sleep(self.monitoring_interval_in_seconds)
        print()

    def __plot_graph(self):
        plt.figure(figsize=(12, 6))

        # Plot CPU Usage
        plt.subplot(3, 1, 1)
        self.__plot_experiment("CPU", self.forecasting.model_cpu)

        # Plot Memory Usage
        plt.subplot(3, 1, 2)
        self.__plot_experiment("Mem", self.forecasting.model_mem)

        # Plot Disk Usage
        plt.subplot(3, 1, 3)
        self.__plot_experiment("Disk", self.forecasting.model_disk)

        plt.tight_layout()

        if self.save_plot:
            path_to_save = self.filename.replace(".csv", ".png")
            plt.savefig(path_to_save, dpi=300)

        plt.show()

    def __plot_experiment(self, resource: str, model):
        x_train = self.forecasting.train_data["Timestamp"]
        x_test = self.forecasting.test_data["Timestamp"]
        training_end = x_train.iloc[-1]

        plt.plot(
            x_train,
            self.forecasting.train_data[resource],
            label=f"Train {resource} Usage",
            color="g",
        )
        plt.plot(
            x_test,
            self.forecasting.test_data[resource],
            label=f"Test {resource} Usage",
            color="b",
        )

        plt.plot(
            x_train,
            model.predict(pd.to_numeric(x_train).values.reshape(-1, 1)),
            label=f"Train {resource} Prediction",
            color="r",
        )
        plt.plot(
            x_test,
            model.predict(pd.to_numeric(x_test).values.reshape(-1, 1)),
            label=f"Test {resource} Prediction",
            color="orange",
        )
        plt.axvline(training_end, color="gray", linestyle="--")

        plt.legend(loc="upper left", fontsize="small")
        plt.xlabel("Timestamp")
        plt.ylabel("Usage (%)")
        plt.title(f"{resource} Usage and Prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resource Monitoring and Prediction CLI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear_regression",
        choices=["linear_regression"],
        help="Model for time series prediction",
    )
    parser.add_argument(
        "--monitoring-time-in-seconds",
        type=int,
        default=60,
        help="Time in seconds to monitor the resource usage",
    )
    parser.add_argument(
        "--monitoring-interval-in-seconds",
        type=int,
        default=1,
        help="Interval between each monitoring in seconds",
    )
    parser.add_argument(
        "--prediction-interval-in-seconds",
        type=int,
        default=60,
        help="Interval in seconds to predict the resource usage",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="/home/gabriel/Repositories/software-aging-framework/data.csv",
        help="Path to save the monitoring data",
    )
    parser.add_argument(
        "--save-plot",
        type=bool,
        default=False,
        help="Save the plot as a png file",
    )
    parser.add_argument(
        "--run-in-real-time",
        type=bool,
        default=False,
        help="Run the monitoring and prediction in real time",
    )
    args = parser.parse_args()

    framework = Framework(**vars(args))
    framework.run()
