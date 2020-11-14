import numpy as np
from subroutines import *
import matplotlib.pyplot as plt
stock_data = load_csv("GBPUSD_H1_202001020000_202011102200")

n = 1
OPEN = [stock_data[i]["OPEN"]/stock_data[i-1]["OPEN"] for i in range(1, len(stock_data))]
plt.plot(OPEN)
plt.show()

while n<=stock_data.__len__():
    data, n = feeder(stock_data, n)
