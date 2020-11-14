
def load_csv(csv_name):
    filepath = r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords"
    f = open(filepath + f"\{csv_name}.csv", "r").read().replace("\t", ",")
    data = [i.split(",") for i in f.split("\n")[1:-1]]
    information = []
    for i in data:
        information.append({"DATE": i[0], "TIME": i[1], "OPEN": float(i[2]), "HIGH": float(i[3]), "LOW": float(i[4]), "CLOSE": float(i[5])})
    return information


def feeder(stock_data, n):
    return stock_data[:n], n+1