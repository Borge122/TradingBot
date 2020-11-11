import os
import pickle
#accuracy_list = {"FCL-Sig": [], "FCL-gSig": [], "FCL-Tanh": [], "CONV_relu_Sig": [], "CONV_mrelu_Sig": [], "CONV_mrelu_gSig": [], "CONV_relu_gSig": []}
#pickle.dump(accuracy_list, open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\Data\Data.pickle", "wb"))

for i in range(10):
    print(f"ROUND {i}")
    os.system("python FunctionTesting.py")
print("DONE!")