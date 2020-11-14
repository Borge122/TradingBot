import matplotlib.pyplot as plt
import pickle
import numpy as np
accuracy_list = dict(pickle.load(open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\Data\Data.pickle", "rb")))

experiment = "FCL-gSig"
mean = np.mean(np.array(accuracy_list[experiment]), axis=0)
std = np.std(np.array(accuracy_list[experiment]), axis=0)
plt.plot(mean, color=[1, 0.5, 0.5], label="gSig")
plt.fill_between(np.arange(0, len(mean)), mean-std, mean+std, color=[1, 0.5, 0.5, 0.2])
plt.plot(mean+std, linestyle='dashed', color=[1, 0.5, 0.5])
plt.plot(mean-std, linestyle='dashed', color=[1, 0.5, 0.5])
plt.plot(accuracy_list[experiment][0], color=[1, 0.5, 0.5 ,0.1], label="gSig")
for j in accuracy_list[experiment][1:]:
    plt.plot(j, color=[1, 0.5, 0.5 ,0.1])


experiment = "FCL-Tanh"
mean = np.mean(np.array(accuracy_list[experiment]), axis=0)
std = np.std(np.array(accuracy_list[experiment]), axis=0)
plt.plot(mean, color=[0.5, 1, 0.5], label="Tanh")
plt.fill_between(np.arange(0, len(mean)), mean-std, mean+std, color=[0.5, 1, 0.5, 0.2])
plt.plot(mean+std, linestyle='dashed', color=[0.5, 1, 0.5])
plt.plot(mean-std, linestyle='dashed', color=[0.5, 1, 0.5])
plt.plot(accuracy_list[experiment][0], color=[0.5, 1, 0.5 ,0.1], label="Tanh")
for j in accuracy_list[experiment][1:]:
    plt.plot(j, color=[0.5, 1, 0.5 ,0.1])


experiment = "FCL-Sig"
mean = np.mean(np.array(accuracy_list[experiment]), axis=0)
std = np.std(np.array(accuracy_list[experiment]), axis=0)
plt.plot(mean, color=[0.5, 0.5, 1], label="Sig")
plt.fill_between(np.arange(0, len(mean)), mean-std, mean+std, color=[0.5, 0.5, 1, 0.2])
plt.plot(mean+std, linestyle='dashed', color=[0.5, 0.5, 1])
plt.plot(mean-std, linestyle='dashed', color=[0.5, 0.5, 1])
plt.plot(accuracy_list[experiment][0], color=[0.5, 0.5, 1 ,0.1], label="Sig")
for j in accuracy_list[experiment][1:]:
    plt.plot(j, color=[0.5, 0.5, 1, 0.1])


experiment = "FCL-gSig-Cost-1,1"
mean = np.mean(np.array(accuracy_list[experiment]), axis=0)
std = np.std(np.array(accuracy_list[experiment]), axis=0)
plt.plot(mean, color=[1, 1, 0.5], label="FCL-gSig-Cost-1,1")
plt.fill_between(np.arange(0, len(mean)), mean-std, mean+std, color=[1, 1, 0.5, 0.2])
plt.plot(mean+std, linestyle='dashed', color=[1, 1, 0.5])
plt.plot(mean-std, linestyle='dashed', color=[1, 1, 0.5])
plt.plot(accuracy_list[experiment][0], color=[1, 1, 0.5 ,0.1], label="FCL-gSig-Cost-1,1")
for j in accuracy_list[experiment][1:]:
    plt.plot(j, color=[1, 1, 0.5 ,0.1])

plt.legend()
plt.show()