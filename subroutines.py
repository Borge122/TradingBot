import datetime


def load_csv(csv_name):
    filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"
    f = open(filepath + f"\{csv_name}.csv", "r").read().replace("\t", ",")
    data = [i.split(",") for i in f.split("\n")[1:-1]]
    information = []
    for i in data:

        information.append({"DATETIME": datetime.datetime.strptime(i[0]+";"+i[1], "%Y.%m.%d;%H:%M:%S"), "OPEN": float(i[2]), "HIGH": float(i[3]), "LOW": float(i[4]), "CLOSE": float(i[5])})
    return information


def load_csv_time_consistent(csv_name, time_logs):
    print("START")
    filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"
    f = open(filepath + f"\{csv_name}.csv", "r").read().replace("\t", ",")
    data = [i.split(",") for i in f.split("\n")[10:-10]]

    for i in data:
        time_logs[datetime.datetime.strptime(i[0]+";"+i[1], "%Y.%m.%d;%H:%M:%S")] = {"OPEN": float(i[2]), "HIGH": float(i[3]), "LOW": float(i[4]), "CLOSE": float(i[5])}
    return time_logs

def feeder(stock_data, n):
    return stock_data[:n], n+1