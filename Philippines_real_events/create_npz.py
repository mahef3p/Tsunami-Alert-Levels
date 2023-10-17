import numpy as np
from sklearn.model_selection import train_test_split

X = np.loadtxt('X.txt', dtype=np.float32)
Y = np.loadtxt('Y.txt', dtype=np.float32)

num_ejemplos = 32000
num_ejemplos_train = 24000
X = X[0:num_ejemplos,:]
Y = Y[0:num_ejemplos,:]

print('Shape X: ', X.shape)
print('Shape Y: ', Y.shape)

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=num_ejemplos-num_ejemplos_train)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
del X,Y
print('X, Samples train, val, test: {}, {}, {}'.format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
print('Y, Samples train, val, test: {}, {}, {}'.format(Y_train.shape[0], Y_val.shape[0], Y_test.shape[0]))

nclases_train = np.sum(Y_train, axis=0)
nclases_val = np.sum(Y_val, axis=0)
nclases_test = np.sum(Y_test, axis=0)
print('Classes train: [', nclases_train[0], ',', nclases_train[1], ',', nclases_train[2], ',', nclases_train[3], ']')
print('Classes val: [', nclases_val[0], ',', nclases_val[1], ',', nclases_val[2], ',', nclases_val[3], ']')
print('Classes test: [', nclases_test[0], ',', nclases_test[1], ',', nclases_test[2], ',', nclases_test[3], ']')

# Save the sets in sets.npz file
np.savez('sets.npz', X_train=X_train, X_val=X_val, X_test=X_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test)

