import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./2.1.csv")


convert_dict = {'time' : float}
data = data.astype(convert_dict)
# print(data.dtypes)
# plt.plot(data["threads"], data["time"].values, label="Time (s)")
# plt.plot(data["threads"], data["usr_time"], label="User Time (s)")
# plt.plot(data["threads"], data["sys_time"], label="System Time (s)")
plt.plot(data["threads"], data["cycles"], label="Cycles")
plt.legend()
plt.grid(True)
plt.ylabel("Number of Cycles for 10 repetitions")
plt.xlabel("Number of Threads")
plt.title("Perf Stat")
plt.savefig("graph-2.png")
# plt.xlabel()
# plt.show()