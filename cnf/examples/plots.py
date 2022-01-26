import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv("experiments_logs/experiment_2022-01-13T21:19:02.630144.log")
print(df.columns)

x = df.dim
y = df.avg_train_time_sec
degree = 1
y_ = np.polyval(p=np.polyfit(x=x, y=y, deg=degree), x=x)
print(math.sqrt(mean_squared_error(y_,y)))
plt.title('dimension vs running-time')
plt.xlabel('dimension')
plt.ylabel('running-time in secs')
plt.plot(x, y, "-b")
plt.plot(x, y_, "-r")
plt.legend(["raw-data",f"polyfit with degree ={degree}"])
plt.savefig('fig1.png')
plt.clf()

y = df.out_of_sample_loss
degree = 5
y_ = np.polyval(p=np.polyfit(x=x, y=y, deg=degree), x=x)
print(math.sqrt(mean_squared_error(y_, y)))
plt.title('dimension vs average-loss(-LL)')
plt.xlabel('dimension')
plt.ylabel('avg-loss')
plt.plot(x, y, "-b")
plt.plot(x, y_, "-r")
plt.legend(["raw-data", f"poly-fit with degree = {degree}"])
plt.savefig('fig2.png')
