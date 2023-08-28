import time
from datetime import datetime
from multiprocessing import Process, Queue

import pandas as pd
import psutil


class ResourceMonitor:
    class ProcessNotFound(Exception):
        pass

    def __init__(self, interval_in_seconds: int, process_name: str, filename: str):
        self.interval_in_seconds = interval_in_seconds
        self.filename = filename
        self.process_name = process_name
        self.process = None

    def monitor(self):
        self.process = self.__get_process()
        self.__create_file()

        while True:
            cpu_percent = self.process.cpu_percent()
            mem_info = self.process.memory_info()
            mem_usage = mem_info.rss / (1024**1)  # Memory usage in KB
            # TODO: monitor the process disk usage
            disk_usage = psutil.disk_usage("/").used / (1024 * 1)  # Disk usage in in KB

            timestamp = datetime.now()
            data = (timestamp, cpu_percent, mem_usage, disk_usage)

            dataframe = pd.DataFrame(
                [data], columns=["Timestamp", "CPU", "Mem", "Disk"]
            )
            dataframe.to_csv(self.filename, mode="a", index=False, header=False)

            time.sleep(self.interval_in_seconds)

    def __get_process(self):
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if self.process_name.lower() in process.info["name"].lower():
                return psutil.Process(process.info["pid"])
        raise self.ProcessNotFound(f"Process '{self.process_name}' not found.")

    def __create_file(self):
        dataframe = pd.DataFrame(columns=["Timestamp", "CPU", "Mem", "Disk"])
        dataframe.to_csv(self.filename, index=False)


class ResourceMonitorProcess(Process):
    def __init__(
        self, interval_in_seconds: int, process_name: str, filename: str, queue: Queue
    ):
        super(ResourceMonitorProcess, self).__init__()
        self.resource_monitor = ResourceMonitor(
            interval_in_seconds, process_name, filename
        )
        self.queue = queue

    def run(self):
        try:
            self.resource_monitor.monitor()
        except Exception as e:
            self.queue.put(str(e))
            raise e
