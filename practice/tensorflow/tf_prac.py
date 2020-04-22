import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import platform

os = platform.system()
if os.lower() == "windows":
    dataframe = pd.read_csv("..\..\data\pulsar_stars.csv")
else:
    dataframe = pd.read_csv("../../data/pulsar_stars.csv")
dataframe.head()

train, test = train_test_split(dataframe, test_size=0.3)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#tf.keras.backend.set_floatx('float64')

def df_to_dataset(dataframe, shuffle=True, batch_size=64):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target_class')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 5 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of targets:', label_batch )

# We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(train_ds))[0]
# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

meanProf = feature_column.numeric_column(' Mean of the integrated profile')
demo(meanProf)
stDev = feature_column.numeric_column(' Standard deviation of the integrated profile')
demo(stDev)
exKurt = feature_column.numeric_column(' Excess kurtosis of the integrated profile')
demo(exKurt)
skew = feature_column.numeric_column(' Skewness of the integrated profile')
demo(skew)
meanDM = feature_column.numeric_column(' Mean of the DM-SNR curve')
demo(meanDM)
stDevDM = feature_column.numeric_column(' Standard deviation of the DM-SNR curve')
demo(stDevDM)
exKurtDM = feature_column.numeric_column(' Excess kurtosis of the DM-SNR curve')
demo(exKurtDM)
skewDM = feature_column.numeric_column(' Skewness of the DM-SNR curve')
demo(skewDM)
feature_columns = []
# numeric cols
for header in ['meanProf', 'stDev', 'exKurt', 'skew', 'meanDM', 'stDevDM', 'exKurtDM', 'skewDM']:
  feature_columns.append(feature_column.numeric_column(header))


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 64
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.fit(train_ds,
            #validation_data=val_ds,
            #epochs=5)

#loss, accuracy = model.evaluate(test_ds)
#print("Accuracy", accuracy)
