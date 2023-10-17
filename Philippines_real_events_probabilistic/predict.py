import numpy as np
import tensorflow as tf
from tensorflow import keras

# Put 1975 or 1995
X = np.loadtxt('X_1975.txt', dtype=np.float32)

print('Shape X: ', X.shape)

# Normalize inputs in [0,1]
X[:,0] = (X[:,0] - 125.477778)/0.6  # lon
X[:,1] = (X[:,1] - 11.994444)/0.9   # lat
X[:,2] = (X[:,2] - 20.0)/30.0       # depth
X[:,3] = (X[:,3] - 40.0)/100.0      # fault length
X[:,4] = (X[:,4] - 30.0)/40.0       # fault width
X[:,5] = (X[:,5] - 145.0)/10.0      # strike
X[:,6] = (X[:,6] - 29.0)/10.0       # dip
X[:,7] = (X[:,7] - 80.0)/20.0       # rake
X[:,8] = (X[:,8] - 0.2)/9.8         # slip

print('Range X: [{}, {}]'.format(np.min(X), np.max(X)))

model = keras.models.load_model("modelo.h5")

Y_pred = model.predict(X)
media = np.mean(Y_pred, axis=0)
desv = np.std(Y_pred, axis=0)

print('Mean:')
print(media)
print('Std. deviation:')
print(desv)

