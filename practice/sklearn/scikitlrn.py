import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

os = platform.system()
if os.lower() == "windows":
    data = pd.read_csv("..\..\data\pulsar_stars.csv")
else:
    data = pd.read_csv("../../data/pulsar_stars.csv")

clf = RandomForestClassifier(random_state=10)
data = np.array(data)
X = data[:, 0:8]
Y = data[:, 8]
print(data.shape, X.shape, Y.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#print(all(Y == prediction))

#false_count = 0
#for i in range(Y_test.shape[0]):
#    if prediction[i] != Y[i]:
#        false_count += 1

#print((false_count/Y.shape[0])*100)
