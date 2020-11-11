import os
import datetime
filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"
paths = [i for i in os.listdir(filepath)]
f = open(filepath+f"\{paths[0]}", "r").read().replace("\t", ", ")
open(filepath+f"\{paths[0]}-edited", "w").write(f)
print(f)
#2020.01.02	01:00:00	1.08409	1.08570	1.08409	1.08564	