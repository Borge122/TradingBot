import datetime

def load_csv(csv_name):
    filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"
    f = open(filepath + f"\{csv_name}.csv", "r").read().replace("\t", ",")
    data = [i.split(",") for i in f.split("\n")[1:-1]]
    information = []
    for i in data:

        information.append({"DATETIME": datetime.datetime.strptime(i[0]+";"+i[1], "%Y.%m.%d;%H:%M:%S"), "OPEN": float(i[2]), "HIGH": float(i[3]), "LOW": float(i[4]), "CLOSE": float(i[5])})
    return information

def load_csv_time_consistent(csv_name, start_time, end_time):
    filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"
    f = open(filepath + f"\{csv_name}.csv", "r").read().replace("\t", ",")
    data = [i.split(",") for i in f.split("\n")[1:-1]]
    information = [{"DATETIME": datetime.datetime.strptime(start_time, "%Y.%m.%d;%H:%M:%S")}]
    for i in data:
        while not datetime.datetime.strptime(i[0]+";"+i[1], "%Y.%m.%d;%H:%M:%S") == information[-1]["DATETIME"]:
            information.append({"DATETIME": information[-1]["DATETIME"]+datetime.timedelta(minutes=20)})
        information[-1] = {"DATETIME": datetime.datetime.strptime(i[0]+";"+i[1], "%Y.%m.%d;%H:%M:%S"), "OPEN": float(i[2]), "HIGH": float(i[3]), "LOW": float(i[4]), "CLOSE": float(i[5])}
    while datetime.datetime.strptime(end_time, "%Y.%m.%d;%H:%M:%S") > information[-1]["DATETIME"]:
        information.append({"DATETIME": information[-1]["DATETIME"] + datetime.timedelta(minutes=20)})
    return information

def feeder(stock_data, n):
    return stock_data[:n], n+1