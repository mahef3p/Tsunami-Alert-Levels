import numpy as np
import tensorflow as tf
from tensorflow import keras

num_samples = 2
X = np.zeros((num_samples,9))
Y = np.array([[0,1,0,0], [1,0,0,0]])

print('Shape X: ', X.shape)
print('Shape Y: ', Y.shape)

# 1975 tsunami (Mw=7.6)
X[0,0] = 125.993
X[0,1] =  12.540
X[0,2] =  50.0
X[0,3] =  97.72
X[0,4] =  53.21
X[0,5] = 150.0
X[0,6] =  34.0
X[0,7] =  95.0
X[0,8] =   2.03

# 1995 tsunami (Mw=7.1)
X[1,0] = 125.580
X[1,1] =  12.059
X[1,2] =  21.0
X[1,3] =  51.88
X[1,4] =  37.24
X[1,5] = 150.0
X[1,6] =  34.0
X[1,7] =  95.0
X[1,8] =   0.97

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
clases_pred = np.argmax(Y_pred, axis=1)
clases_ok = np.argmax(Y, axis=1)
naciertos = np.sum(clases_pred==clases_ok)
porcentaje_aciertos = naciertos*100.0/num_samples

print("Predictions:")
print(Y_pred)
print("Correct predictions: %d" % (naciertos))

