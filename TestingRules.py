import numpy as np
import matplotlib.pyplot as plt
filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"


def load_csv(csv_name):
    f = open(filepath + f"\{csv_name}.csv", "r").read().replace("\t", ",")
    data = [i.split(",") for i in f.split("\n")[1:-1]]
    information = []
    for i in data:
        information.append({"DATE": i[0], "TIME": i[1], "OPEN": float(i[2]), "HIGH": float(i[3]), "LOW": float(i[4]), "CLOSE": float(i[5])})
    return information

data_from_csv = load_csv("GBPUSD_M12_202001020000_202011102300")
for i in data_from_csv:
    print(i)


# #Acts as a simulator, pretending the data is coming in live:
iterative_value = 1
colour = []
while iterative_value < 2000: # len(data_from_csv):
    colour.append(0)
    data = data_from_csv[:iterative_value]
    OPEN_DIFFERENTIAL = [(np.sign(data[i]["OPEN"]-data[i-1]["OPEN"])+1)/2 for i in range(1, len(data))]
    CLOSE_DIFFERENTIAL = [(np.sign(data[i]["CLOSE"]-data[i-1]["CLOSE"])+1)/2 for i in range(1, len(data))]

    # #PHASE 1
    p1_percentage_trending_upwards = 0.7 # 0.7
    p1_previous_hours_to_check = 5 # 6
    # #PHASE 2
    p2_percentage_trending_downwards = 0.4 # 0.4
    p2_previous_hours_to_check = 2 # 3

    if np.average(OPEN_DIFFERENTIAL[-p2_previous_hours_to_check:]) < p2_percentage_trending_downwards and np.average(CLOSE_DIFFERENTIAL[-p2_previous_hours_to_check:]) < p2_percentage_trending_downwards and \
            np.average(OPEN_DIFFERENTIAL[-p1_previous_hours_to_check-p2_previous_hours_to_check:-p2_previous_hours_to_check]) > p1_percentage_trending_upwards and np.average(CLOSE_DIFFERENTIAL[-p1_previous_hours_to_check-p2_previous_hours_to_check:-p2_previous_hours_to_check]) > p1_percentage_trending_upwards:
        if colour.__len__() > p2_previous_hours_to_check:
            for k in range(p2_previous_hours_to_check):
                colour[-k-1] = 2.
        if colour.__len__() > p1_previous_hours_to_check+p2_previous_hours_to_check:
            for k in range(p1_previous_hours_to_check):
                colour[-k-p2_previous_hours_to_check-1] = 1.
    iterative_value += 1

flag = False
for i in range(1, len(data)):
    if flag:
        flag = False
        plt.plot([i - 1, i], [data[i - 1]["OPEN"], data[i]["OPEN"]], color=[1, 1, 0])
    else:
        if colour[i] == 0:
            plt.plot([i-1, i], [data[i-1]["OPEN"], data[i]["OPEN"]], color=[0, 0, 0])
            #plt.scatter(i, data[i]["CLOSE"], color=[0, 0, 0], s=1)
        elif colour[i] == 1:
            plt.plot([i-1, i], [data[i-1]["OPEN"], data[i]["OPEN"]], color=[1, 0, 0])
            #plt.scatter(i, data[i]["CLOSE"], color=[0, 0, 0], s=1)
        elif colour[i] == 2:
            plt.plot([i-1, i], [data[i-1]["OPEN"], data[i]["OPEN"]], color=[0, 1, 0])
            flag = True

plt.show()