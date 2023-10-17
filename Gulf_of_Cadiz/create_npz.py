import numpy as np
from sklearn.model_selection import train_test_split

X = np.loadtxt('X.txt', dtype=np.float32)
Y_cadiz = np.loadtxt('Y_cadiz.txt', dtype=np.float32)
Y_rota = np.loadtxt('Y_rota.txt', dtype=np.float32)

num_ejemplos = 128000
num_ejemplos_train = 96000
X = X[0:num_ejemplos,:]
Y_cadiz = Y_cadiz[0:num_ejemplos,:]
Y_rota = Y_rota[0:num_ejemplos,:]

print('Shape X: ', X.shape)
print('Shape Y_cadiz: ', Y_cadiz.shape)
print('Shape Y_rota: ', Y_rota.shape)

# Normalize inputs in [0,1]
X[:,0] = (X[:,0] + 8.2)/1.6     # lon
X[:,1] = (X[:,1] - 36.0)/0.9    # lat
X[:,2] = (X[:,2] - 5.0)/10.0    # depth
X[:,3] = (X[:,3] - 25.0)/60.0   # fault length
X[:,4] = (X[:,4] - 20.0)/50.0   # fault width
X[:,5] = (X[:,5] + 105.0)/10.0  # strike
X[:,6] = (X[:,6] - 30.0)/10.0   # dip
X[:,7] = (X[:,7] - 80.0)/20.0   # rake
X[:,8] = (X[:,8] - 1.0)/7.0     # slip

print('Range X: [{}, {}]'.format(np.min(X), np.max(X)))
X_train, X_test, Y_cadiz_train, Y_cadiz_test, Y_rota_train, Y_rota_test = train_test_split(X, Y_cadiz, Y_rota, test_size=num_ejemplos-num_ejemplos_train)
X_val, X_test, Y_cadiz_val, Y_cadiz_test, Y_rota_val, Y_rota_test = train_test_split(X_test, Y_cadiz_test, Y_rota_test, test_size=0.5)
del X,Y_cadiz,Y_rota
print('X,       Samples train, val, test: {}, {}, {}'.format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))
print('Y_cadiz, Samples train, val, test: {}, {}, {}'.format(Y_cadiz_train.shape[0], Y_cadiz_val.shape[0], Y_cadiz_test.shape[0]))
print('Y_rota,  Samples train, val, test: {}, {}, {}'.format(Y_rota_train.shape[0], Y_rota_val.shape[0], Y_rota_test.shape[0]))

nclases_train = np.sum(Y_cadiz_train, axis=0)
nclases_val = np.sum(Y_cadiz_val, axis=0)
nclases_test = np.sum(Y_cadiz_test, axis=0)
print('Classes Cadiz train: [', nclases_train[0], ',', nclases_train[1], ',', nclases_train[2], ',', nclases_train[3], ']')
print('Classes Cadiz val: [', nclases_val[0], ',', nclases_val[1], ',', nclases_val[2], ',', nclases_val[3], ']')
print('Classes Cadiz test: [', nclases_test[0], ',', nclases_test[1], ',', nclases_test[2], ',', nclases_test[3], ']')

nclases_train = np.sum(Y_rota_train, axis=0)
nclases_val = np.sum(Y_rota_val, axis=0)
nclases_test = np.sum(Y_rota_test, axis=0)
print('Classes Rota train: [', nclases_train[0], ',', nclases_train[1], ',', nclases_train[2], ',', nclases_train[3], ']')
print('Classes Rota val: [', nclases_val[0], ',', nclases_val[1], ',', nclases_val[2], ',', nclases_val[3], ']')
print('Classes Rota test: [', nclases_test[0], ',', nclases_test[1], ',', nclases_test[2], ',', nclases_test[3], ']')

# Save the sets in sets.npz file
np.savez('sets.npz', X_train=X_train, X_val=X_val, X_test=X_test,\
Y_cadiz_train=Y_cadiz_train, Y_cadiz_val=Y_cadiz_val, Y_cadiz_test=Y_cadiz_test,\
Y_rota_train=Y_rota_train, Y_rota_val=Y_rota_val, Y_rota_test=Y_rota_test)

