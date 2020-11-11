from TestingRules import *
#exp moving avg
import numpy as np
import matplotlib.pyplot as plt
import datetime
load_csv("EURCHF_H1_202001020100_202011101800.csv")

data = [
{'DATE': '2020.11.09', 'TIME': '01:00:00', 'OPEN': 1.06893, 'HIGH': 1.06948, 'LOW': 1.06847, 'CLOSE': 1.06945},
{'DATE': '2020.11.09', 'TIME': '02:00:00', 'OPEN': 1.06945, 'HIGH': 1.06975, 'LOW': 1.06888, 'CLOSE': 1.06958},
{'DATE': '2020.11.09', 'TIME': '03:00:00', 'OPEN': 1.06958, 'HIGH': 1.06988, 'LOW': 1.06949, 'CLOSE': 1.06964},
{'DATE': '2020.11.09', 'TIME': '04:00:00', 'OPEN': 1.06963, 'HIGH': 1.06978, 'LOW': 1.06959, 'CLOSE': 1.06971},
{'DATE': '2020.11.09', 'TIME': '05:00:00', 'OPEN': 1.0697, 'HIGH': 1.06984, 'LOW': 1.06954, 'CLOSE': 1.06959},
{'DATE': '2020.11.09', 'TIME': '06:00:00', 'OPEN': 1.06961, 'HIGH': 1.06975, 'LOW': 1.06929, 'CLOSE': 1.06933},
{'DATE': '2020.11.09', 'TIME': '07:00:00', 'OPEN': 1.06933, 'HIGH': 1.0697, 'LOW': 1.06891, 'CLOSE': 1.06894},
{'DATE': '2020.11.09', 'TIME': '08:00:00', 'OPEN': 1.06894, 'HIGH': 1.06938, 'LOW': 1.06891, 'CLOSE': 1.06898},
{'DATE': '2020.11.09', 'TIME': '09:00:00', 'OPEN': 1.06896, 'HIGH': 1.06941, 'LOW': 1.06848, 'CLOSE': 1.06876},
{'DATE': '2020.11.09', 'TIME': '10:00:00', 'OPEN': 1.06877, 'HIGH': 1.06904, 'LOW': 1.06824, 'CLOSE': 1.06863},
{'DATE': '2020.11.09', 'TIME': '11:00:00', 'OPEN': 1.06864, 'HIGH': 1.06904, 'LOW': 1.06799, 'CLOSE': 1.06893},
]
#time = datetime.datetime.strptime(listed[0]["DATE"]+";"+listed[0]["TIME"], "%Y.%M.%d;%H:%M:%S")
#listed[0]["TIME"].split(":")[0]
#listed[4]["OPEN"]

data[0]["TIME"].split(":")[-1]


def expmovingavg(period):
    t = 1
    ema = [data[0]["CLOSE"]]
    while t < data[0]["TIME"].split(":")[-1] :

        ema_new = (data[t]["CLOSE"] * ema[t])*(2/(period+1)) + ema[t]
        ema.append(ema_new)
        t = t + 1

    return ema

time = data[:]["TIME"].split(":")[-1]


#time = [datapoint["TIME"][:2] for datapoint in data]
#print(time)
#raise 
plt.plot(expmovingavg(20))
plt.show()
