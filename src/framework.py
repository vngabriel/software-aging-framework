import sys
import time

import pandas as pd
import yaml
from matplotlib import pyplot as plt

from src.forecasting import Forecasting
from src.monitor import ResourceMonitorProcess
from src.utils import normalize, denormalize


class Framework:
    def __init__(
        self,
        run_monitoring: bool,
        resources_to_predict: list[str],
        monitoring_time_in_seconds: int,
        monitoring_interval_in_seconds: int,
        filename: str,
        model: str,
        save_plot: bool,
        run_in_real_time: bool,
        process_name: str,
        memory_threshold: float,
        cpu_threshold: float,
        disk_threshold: float,
    ):
        self.run_monitoring = run_monitoring
        self.resources = resources_to_predict
        self.monitoring_time_in_seconds = monitoring_time_in_seconds
        self.monitoring_interval_in_seconds = monitoring_interval_in_seconds
        self.filename = filename
        self.model_name = model
        self.save_plot = save_plot
        self.run_in_real_time = run_in_real_time
        self.process_name = process_name
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
        self.forecasting: Forecasting | None = None
        self.monitor_process: ResourceMonitorProcess | None = None
        if self.run_monitoring:
            self.monitor_process = ResourceMonitorProcess(
                self.monitoring_interval_in_seconds, self.process_name, self.filename
            )

    def run(self):
        if self.run_in_real_time:
            self.__run_real_time()
        else:
            self.__run_experiment()

    def __run_monitoring(self):
        self.monitor_process.start()
        self.__countdown()
        self.monitor_process.terminate()

    def __run_experiment(self):
        if self.run_monitoring:
            self.__run_monitoring()

        dataframe = pd.read_csv(self.filename)

        self.forecasting = Forecasting(dataframe, self.model_name, self.resources)
        self.forecasting.train()
        self.__plot_graph()

    def __run_real_time(self):
        if self.run_monitoring:
            self.monitor_process.start()
            self.__countdown()

        dataframe = pd.read_csv(self.filename)

        self.forecasting = Forecasting(dataframe, self.model_name, self.resources)
        self.forecasting.train()

        plt.ion()  # turn on interactive mode for real-time plotting
        plt.figure(figsize=(10, 6))

        while True:
            plt.clf()  # clear the current figure

            # collect real-time monitoring data
            current_data = pd.read_csv(self.filename)
            current_data = current_data[self.resources]

            n_steps = 2
            n_seq = 2
            normalization_params = {}

            for resource in self.resources:
                current_data[resource], s_min, s_max = normalize(current_data[resource])
                normalization_params[resource] = (s_min, s_max)

            # the last 4 rows of the current data are used for forecasting (n_steps = 4 or n_seq = 2 and n_steps = 2)
            reshaped_current_data = current_data[-4:].values.reshape(
                (1, n_seq, 1, n_steps, len(self.resources))
            )

            # perform forecasting using the trained model
            predictions = self.forecasting.predict(reshaped_current_data)

            flag_list = []
            thresholds_by_resource = {
                "Mem": self.memory_threshold,
                "CPU": self.cpu_threshold,
                "Disk": self.disk_threshold,
            }

            # compare predictions with thresholds and update flag_list and plot the results
            for idx, resource in enumerate(self.resources):
                s_min, s_max = normalization_params[resource]
                denormalized_predictions = denormalize(
                    predictions[:, idx], s_min, s_max
                )

                resource_value = denormalized_predictions[0]
                if resource_value > thresholds_by_resource[resource]:
                    flag_list.append(1)
                else:
                    flag_list.append(0)

                s_min, s_max = normalization_params[resource]
                denormalized_predictions = denormalize(
                    predictions[:, idx], s_min, s_max
                )

                plt.subplot(len(self.resources), 1, idx + 1)
                plt.subplots_adjust(hspace=0.8)

                plt.plot(
                    current_data.values[:, idx],
                    label=f"Real-Time Data ({resource})",
                    color="blue",
                )
                plt.plot(
                    current_data.shape[0] + 1,
                    denormalized_predictions[0],
                    marker="o",
                    color="red",
                    label=f"Forecasted Value ({resource})",
                )

                plt.legend(loc="upper left", fontsize="small")
                plt.xlabel("Time")
                plt.ylabel("Resource Usage")
                plt.title(f"Real-Time {resource} Usage and Forecasting")

            # check if rejuvenation should be triggered
            if flag_list.count(1) > 0:
                print("\nActivated Rejuvenation\n")
                print("Flag list:", flag_list)
                # TODO: trigger rejuvenation

            plt.draw()
            plt.pause(0.01)

        plt.ioff()
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
        self.forecasting.plot_results()

        if self.save_plot:
            path_to_save = self.filename.replace(".csv", ".png")
            plt.savefig(path_to_save, dpi=300)

        plt.show()


class FrameworkConfig:
    def __init__(self):
        with open("config.yaml", "r") as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)

        framework = Framework(**config["framework"])
        framework.run()
