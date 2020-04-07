import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

os = platform.system()
if os.lower() == "windows":
    data = pd.read_csv("..\data\pulsar_stars.csv")
else:
    data = pd.read_csv("../data/pulsar_stars.csv")

clf = RandomForestClassifier(random_state=0)
X = data[:, 0:8]
Y = data[:, 8]
print(data.shape, X.shape, Y.shape)

clf.fit(X, Y)
prediction = clf.predict(X)

print(all(Y == prediction))

false_count = 0
for i in range(Y.shape[0]):
    if prediction[i] == Y[i]:
        false_count += 1

print((false_count/Y.shape[0])*100)
