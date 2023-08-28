import subprocess
import sys
import time
from multiprocessing import Queue

import pandas as pd
import psutil
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
        directory_path: str,
        model: str,
        path_to_load_weights: str | None,
        path_to_save_weights: str | None,
        save_plot: bool,
        run_in_real_time: bool,
        process_name: str,
        memory_threshold: float,
        cpu_threshold: float,
        disk_threshold: float,
        number_of_predictions: int,
        start_command: str,
        restart_command: str | None,
    ):
        self.run_monitoring = run_monitoring
        self.resources = resources_to_predict
        self.monitoring_time_in_seconds = monitoring_time_in_seconds
        self.monitoring_interval_in_seconds = monitoring_interval_in_seconds
        self.directory_path = directory_path
        self.model_name = model
        self.path_to_load_weights = path_to_load_weights
        self.path_to_save_weights = path_to_save_weights
        self.save_plot = save_plot
        self.run_in_real_time = run_in_real_time
        self.process_name = process_name
        self.thresholds_by_resource = {
            "Mem": memory_threshold,
            "CPU": cpu_threshold,
            "Disk": disk_threshold,
        }
        self.number_of_predictions = number_of_predictions
        self.start_command = start_command
        self.restart_command = restart_command
        self.forecasting: Forecasting | None = None
        self.monitor_process: ResourceMonitorProcess | None = None
        self.error_queue = Queue()

        if self.run_in_real_time or self.run_monitoring:
            self.path_to_save_weights = self.__create_weights_filename(
                self.path_to_save_weights
            )
            self.filename = self.__create_filename(self.directory_path)
            self.monitor_process = ResourceMonitorProcess(
                self.monitoring_interval_in_seconds,
                self.process_name,
                self.filename,
                self.error_queue,
            )
        else:
            self.filename = self.directory_path

    @staticmethod
    def __create_filename(directory_path: str) -> str:
        current_time = time.strftime("%Y-%m-%d_%H:%M:%S")
        return f"{directory_path}/log_{current_time}.csv"

    @staticmethod
    def __create_weights_filename(directory_path: str | None) -> str | None:
        if directory_path:
            current_time = time.strftime("%Y-%m-%d_%H:%M:%S")
            return f"{directory_path}/log_{current_time}/log_{current_time}"
        return None

    def run(self):
        if self.run_in_real_time:
            self.__run_real_time()
        else:
            self.__run_experiment()

    def __run_experiment(self):
        if self.run_monitoring:
            self.monitor_process.start()

            time.sleep(1)
            if self.error_queue.qsize() > 0:
                print("\nError monitoring process\n")
                return

            self.__countdown()
            self.monitor_process.terminate()

        dataframe = pd.read_csv(self.filename)

        self.forecasting = Forecasting(
            dataframe, self.model_name, self.resources, self.path_to_save_weights
        )
        self.forecasting.train()
        self.__plot_graph()

    def __run_real_time(self):
        self.monitor_process.start()
        time.sleep(1)

        if self.run_monitoring:
            if self.error_queue.qsize() > 0:
                print("\nError monitoring process\n")
                return

            self.__countdown()

            dataframe = pd.read_csv(self.filename)

            if dataframe.shape[0] < 4:
                print(
                    "\nNot enough monitoring data for forecasting, monitor for longer time\n"
                )
                return

            self.forecasting = Forecasting(
                dataframe, self.model_name, self.resources, self.path_to_save_weights
            )
            self.forecasting.train()

        elif self.path_to_load_weights:
            dataframe = pd.read_csv(self.filename)
            self.forecasting = Forecasting(
                dataframe,
                self.model_name,
                self.resources,
                self.path_to_save_weights,
                False,
                self.path_to_load_weights,
            )
        else:
            print(
                "\nUnable to run if monitoring has not been run or model path has not been passed\n"
            )
            self.monitor_process.terminate()
            return

        plt.ion()  # turn on interactive mode for real-time plotting
        plt.figure(figsize=(10, 6))

        running = True
        while running:
            plt.clf()  # clear the current figure

            # collect real-time monitoring data
            current_data = pd.read_csv(self.filename)
            current_data = current_data[self.resources]

            # check if the current data has enough rows for forecasting
            if current_data.shape[0] < 4:
                continue

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
            predictions = self.forecasting.predict_future(
                reshaped_current_data, self.number_of_predictions
            )
            # TODO: save predictions
            # TODO: save plot over time

            flag_list = []

            # compare predictions with thresholds and update flag_list and plot the results
            for idx, resource in enumerate(self.resources):
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
                for i, pred_value in enumerate(denormalized_predictions):
                    if pred_value > self.thresholds_by_resource[resource]:
                        flag_list.append(1)
                    else:
                        flag_list.append(0)

                    plt.plot(
                        current_data.shape[0] + i,
                        pred_value,
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

                for process in psutil.process_iter(attrs=["pid", "name"]):
                    if self.process_name.lower() in process.info["name"].lower():
                        self.__restart_process(
                            process, self.start_command, self.restart_command
                        )
                        running = False
                        break

            plt.draw()
            plt.pause(0.01)

        plt.ioff()
        self.monitor_process.terminate()

    def __restart_process(
        self, process: psutil.Process, start_command: str, restart_command: str | None
    ):
        if restart_command is not None:
            subprocess.Popen(restart_command, shell=True)
        else:
            process.terminate()  # Terminate the process
            process.wait()  # Wait for the process to exit

        # Start the process again
        subprocess.Popen(start_command, shell=True)

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


class FrameworkConfig:
    def __init__(self):
        with open("config.yaml", "r") as yml_file:
            config = yaml.load(yml_file, Loader=yaml.FullLoader)

        framework = Framework(
            **config["general"], **config["monitoring"], **config["real_time"]
        )
        framework.run()
