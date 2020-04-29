import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #splitting data into training and testing subsets
from sklearn.preprocessing import StandardScaler #scaling our features
from sklearn import metrics #analyzing our statistical results

os = platform.system()
if os.lower() == "windows":
    data = pd.read_csv("..\..\data\pulsar_stars.csv") #different syntax on windows and macOS
else:
    data = pd.read_csv("../../data/pulsar_stars.csv")

feature_names = data.columns.values[0:-1]  #our feature feature_names
print(feature_names)

data = np.array(data) #changed to numpy array so we can split the array

X = data[:, 2:3] #only skewness of IP
print(X)
Y = data[:, 8] #our class feauture which is the classifier
print("Data has {} rows and {} columns".format(data.shape[0], data.shape[1]))

clf = RandomForestClassifier(n_estimators=12, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) #splitting data set

#test_size is 0=>1 representing percentage of dataset used for testing and #percentage used for training

sc = StandardScaler() #scaling the features
x_train = sc.fit_transform(x_train) #scaling training features
x_test = sc.transform(x_test) #scaling test features

clf.fit(x_train, y_train)
prediction = clf.predict(x_test)

print('Accuracy:', metrics.accuracy_score(y_test, prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print('Feature importances:', clf.feature_importances_)
