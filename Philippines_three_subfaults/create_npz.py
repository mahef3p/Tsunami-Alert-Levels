import numpy as np
from sklearn.model_selection import train_test_split

X = np.loadtxt('X.txt', dtype=np.float32)
Y = np.loadtxt('Y.txt', dtype=np.float32)

num_ejemplos = 128000
num_ejemplos_train = 96000
X = X[0:num_ejemplos,:]
Y = Y[0:num_ejemplos,:]

print('Shape X: ', X.shape)
print('Shape Y: ', Y.shape)

# Normalize inputs in [0,1]
# PT2
X[:,0]  = (X[:,0] - 0.0)/120.0   # time
X[:,1]  = (X[:,1] - 20.0)/30.0   # depth
X[:,2]  = (X[:,2] - 145.0)/10.0  # strike
X[:,3]  = (X[:,3] - 29.0)/10.0   # dip
X[:,4]  = (X[:,4] - 80.0)/20.0   # rake
X[:,5]  = (X[:,5] - 0.4)/7.6     # slip
# PT3
X[:,6]  = (X[:,6] - 0.0)/120.0   # time
X[:,7]  = (X[:,7] - 20.0)/30.0   # depth
X[:,8]  = (X[:,8] - 160.0)/10.0  # strike
X[:,9]  = (X[:,9] - 40.0)/10.0   # dip
X[:,10] = (X[:,10] - 80.0)/20.0  # rake
X[:,11] = (X[:,11] - 0.4)/7.6    # slip
# PT4
X[:,12] = (X[:,12] - 0.0)/120.0   # time
X[:,13] = (X[:,13] - 20.0)/30.0   # depth
X[:,14] = (X[:,14] - 160.0)/10.0  # strike
X[:,15] = (X[:,15] - 31.0)/10.0   # dip
X[:,16] = (X[:,16] - 80.0)/20.0   # rake
X[:,17] = (X[:,17] - 0.4)/7.6     # slip

print('Range X: [{}, {}]'.format(np.min(X), np.max(X)))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=num_ejemplos-num_ejemplos_train)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5)
del X,Y
print('X, Samples train, val, test: {}, {}, {}'.format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
print('Y, Samples train, val, test: {}, {}, {}'.format(Y_train.shape[0], Y_val.shape[0], Y_test.shape[0]))

nclases_train = np.sum(Y_train, axis=0)
nclases_val = np.sum(Y_val, axis=0)
nclases_test = np.sum(Y_test, axis=0)
print('Classes train: [', nclases_train[0], ',', nclases_train[1], ',', nclases_train[2], ']')
print('Classes val: [', nclases_val[0], ',', nclases_val[1], ',', nclases_val[2], ']')
print('Classes test: [', nclases_test[0], ',', nclases_test[1], ',', nclases_test[2], ']')

# Save the sets in sets.npz file
np.savez('sets.npz', X_train=X_train, X_val=X_val, X_test=X_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test)

