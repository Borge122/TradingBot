from subroutines import *
import numpy as np
import matplotlib.pyplot as plt
"""
This creates a dataset for the neural network to train on. It does require historic data from 20 datasets.
These datasets must start at the same date and time, and have the same time interval of M20. It must be a chronological csv
produced by MetaTrader4 or MetaTrader5.
It is a good idea to include graphs such as US-Oil, FTSE, DAX and DOW-Jones
If you input AUS2CAD then there is no need to also enter CAD2AUS.
"""
meta_trader_outputs = [
    "EURGBP_M20_201501020100_202010302240",
    "EURUSD_M20_201501020100_202010302240",
    "GBPUSD_M20_201501020100_202010302240",
    "USDCHF_M20_201501020100_202010302240",
    "USDJPY_M20_201501020100_202010302240",
]
differential_data = [[(i["CLOSE"]/i["OPEN"])-1 for i in load_csv(set)] for set in meta_trader_outputs]
for i in range(meta_trader_outputs.__len__()):
    print(differential_data[i].__len__())

plt.plot(differential_data[0, -500:])
plt.plot(differential_data[1, -500:])
plt.plot(differential_data[2, -500:])
plt.show()