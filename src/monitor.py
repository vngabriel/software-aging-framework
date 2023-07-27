import time
from datetime import datetime
from multiprocessing import Process

import pandas as pd
import psutil


class ResourceMonitor:
    def __init__(self, interval_in_seconds: int, filename: str):
        self.interval_in_seconds = interval_in_seconds
        self.filename = filename
        self.__create_file()

    def monitor(self):
        while True:
            cpu_percent = psutil.cpu_percent()
            mem_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage("/").percent

            timestamp = datetime.now()
            data = (timestamp, cpu_percent, mem_percent, disk_percent)

            dataframe = pd.DataFrame(
                [data], columns=["Timestamp", "CPU", "Mem", "Disk"]
            )
            dataframe.to_csv(self.filename, mode="a", index=False, header=False)

            time.sleep(self.interval_in_seconds)

    def __create_file(self):
        dataframe = pd.DataFrame(columns=["Timestamp", "CPU", "Mem", "Disk"])
        dataframe.to_csv(self.filename, index=False)


class ResourceMonitorProcess(Process):
    def __init__(self, interval_in_seconds: int, filename: str):
        super(ResourceMonitorProcess, self).__init__()
        self.resource_monitor = ResourceMonitor(interval_in_seconds, filename)

    def run(self):
        self.resource_monitor.monitor()
