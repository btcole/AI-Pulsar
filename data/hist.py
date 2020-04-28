import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./pulsar_stars.csv")
print(data.head())

data = np.array(data)
mean_ip = data[:, 0]
std_ip = data[:, 1]
exkurt_ip = data[:, 2]
skew_ip = data[:, 3]
mean_DMSNR = data[:, 4]
std_DMSNR = data[:, 5]
exkurt_DMSNR = data[:, 6]
skew_DMSNR = data[:, 7]

plt.hist(mean_ip, bins = 100)[2]
plt.title("Histogram of Mean of the Integrated Profile")
plt.savefig('mean_ip.jpg')
plt.show()

plt.hist(std_ip, bins = 100)[2]
plt.title("Histogram of Standard Deviation of the Integrated Profile")
plt.savefig("std_ip.jpg")
plt.show()

plt.hist(exkurt_ip, bins = 100)[2]
plt.title("Histogram of Excess Kurtosis of the Integrated Profile")
plt.savefig("exkurt_ip.jpg")
plt.show()

plt.hist(skew_ip, bins = 100)[2]
plt.title("Histogram of Skewness of the Integrated Profile")
plt.savefig("skew_ip.jpg")
plt.show()

plt.hist(mean_DMSNR, bins = 100)[2]
plt.title("Histogram of Mean of the DM-SNR Profile")
plt.savefig("mean_dmsnr.jpg")
plt.show()

plt.hist(std_DMSNR, bins = 100)[2]
plt.title("Histogram of Standard Deviation of the DM-SNR Profile")
plt.savefig("std_dmsnr.jpg")
plt.show()

plt.hist(exkurt_DMSNR, bins = 100)[2]
plt.title("Histogram of Excess Kurtosis of the DM-SNR Profile")
plt.savefig("exkurt_dmsnr.jpg")
plt.show()

plt.hist(skew_DMSNR, bins = 100)[2]
plt.title("Histogram of Skewness of the DM-SNR Profile")
plt.savefig("skew_dmsnr.jpg")
plt.show()
