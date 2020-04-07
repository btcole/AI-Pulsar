
import platform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os = platform.system()
if os.lower() == "windows":
    data = pd.read_csv("..\data\pulsar_stars.csv")
else:
    data = pd.read_csv("../data/pulsar_stars.csv")
print(data.columns)
print(data)
