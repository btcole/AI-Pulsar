import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #splitting data into training and testing subsets
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

os = platform.system()
if os.lower() == "windows":
    data = pd.read_csv("..\..\data\pulsar_stars.csv")
else:
    data = pd.read_csv("../../data/pulsar_stars.csv")

print(data.head())  #overview of dataset, not scaled well

data = np.array(data) #changed to numpy array so we can split the array

X = data[:, 0:8]
Y = data[:, 8]
print(data.shape, X.shape, Y.shape)

clf = RandomForestClassifier(n_estimators=12, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

"""
test_size is 0=>1 representing percentage of dataset used for testing and percentage used for training
"""

sc = StandardScaler() #scaling the features
x_train = sc.fit_transform(x_train) #
x_test = sc.transform(x_test)

clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

print('Accuracy:', metrics.accuracy_score(y_test, prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print('Feature importances:', clf.feature_importances_)
