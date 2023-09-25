import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


path = "/home/gabrielvn/Faculdade/projetos/software-aging-framework/data"
monitoring_filename = "log_2023-09-12_18-07-45.csv"
predictions_filename = "log_2023-09-12_18-07-45_predictions.csv"


df_monitoring = pd.read_csv(os.path.join(path, monitoring_filename))
df_monitoring = df_monitoring[["CPU", "Mem"]]
df_predictions = pd.read_csv(os.path.join(path, predictions_filename))

monitoring_axis = np.arange(0, len(df_monitoring))
predictions_axis = np.arange(len(df_monitoring) - 200, len(df_monitoring) + (len(df_predictions) * (len(df_predictions.columns) // 2)))
total_points = len(df_monitoring) + len(df_predictions)

print(df_monitoring)
print(df_predictions)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.subplots_adjust(hspace=0.8)

plt.plot(
    df_monitoring["CPU"],
    label="Monitoring Data (CPU)",
    color="blue",
)
j = 0
for col in df_predictions.columns:
    if col.startswith("CPU_n"):
        for i in range(len(df_predictions[col])):
            if i == 0 and j == 0:
                plt.plot(
                    predictions_axis[i],
                    df_predictions[col][i],
                    label="Prediction Data (CPU)",
                    color="red",
                    linestyle="-.",
                    marker='o',
                    markersize=5,
                    markeredgewidth=3,
                )
            else:
                plt.plot(
                    predictions_axis[i],
                    df_predictions[col][i],
                    color="red",
                    linestyle="-.",
                    marker='o',
                    markersize=5,
                    markeredgewidth=3,
                )
        j += 1

plt.legend(loc="upper left", fontsize="small")
plt.xlim(0, total_points + 1000)
plt.ylim(min(df_monitoring["CPU"]) - 1, max(df_monitoring["CPU"]) + 100)
plt.xlabel("Time")
plt.ylabel("Resource Usage")
plt.title("CPU Usage and Prediction")

plt.subplot(2, 1, 2)
plt.subplots_adjust(hspace=0.8)

plt.plot(
    df_monitoring["Mem"],
    label="Monitoring Data (Mem)",
    color="blue",
)
j = 0
for col in df_predictions.columns:
    if col.startswith("Mem_n"):
        for i in range(len(df_predictions[col])):
            if i == 0 and j == 0:
                plt.plot(
                    predictions_axis[i],
                    df_predictions[col][i],
                    label="Prediction Data (Mem)",
                    color="red",
                    linestyle="-.",
                    marker='o',
                    markersize=5,
                    markeredgewidth=3,
                )
            else:
                plt.plot(
                    predictions_axis[i],
                    df_predictions[col][i],
                    color="red",
                    linestyle="-.",
                    marker='o',
                    markersize=5,
                    markeredgewidth=3,
                )
        j += 1

plt.legend(loc="lower left", fontsize="small")
plt.xlim(0, total_points + 1000)
plt.ylim(min(df_monitoring["Mem"]) - 1, max(df_monitoring["Mem"]) + 50000)
plt.xlabel("Time")
plt.ylabel("Resource Usage")
plt.title("Mem Usage and Prediction")
plt.savefig(f"{path}/analysis.png")
# plt.show()
