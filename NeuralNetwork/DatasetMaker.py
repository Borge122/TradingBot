from subroutines import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
"""
This creates a dataset for the neural network to train on. It does require historic data from 20 datasets.
These datasets must start at the same date and time, and have the same time interval of M20. It must be a chronological csv
produced by MetaTrader4 or MetaTrader5.
It is a good idea to include graphs such as US-Oil, FTSE, DAX and DOW-Jones
If you input AUS2CAD then there is no need to also enter CAD2AUS.
"""


meta_trader_outputs = os.listdir(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords")
meta_trader_outputs = [i.replace(".csv", "") for i in meta_trader_outputs]
charts = [load_csv_time_consistent(sets, "2018.01.01;00:00:00", "2020.11.13;22:40:00") for sets in meta_trader_outputs]
pickle.dump(charts, open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords\Charts.pickle", "wb"))
charts = pickle.load(open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords\Charts.pickle", "rb"))

dataset = []
for chart in charts:
    dataset.append([])
    print("New Chart")
    for i in range(len(chart)):
        if "OPEN" in chart[i].keys():
            dataset[-1].append((chart[i]["CLOSE"]/chart[i]["OPEN"])-1)
        else:
            dataset[-1].append(0)

pickle.dump(dataset, open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords\Dataset.pickle", "wb"))
dataset = pickle.load(open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords\Dataset.pickle", "rb"))
for i in dataset:
    print(i.__len__())
