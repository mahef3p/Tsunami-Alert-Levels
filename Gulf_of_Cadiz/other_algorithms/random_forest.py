import numpy as np
import random
import string
import joblib
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Read data sets. Put cadiz or rota
print('Reading X_train, Y_train, X_val, Y_val, X_test, Y_test')
npzfile = np.load('sets.npz')
X_train = npzfile['X_train']
Y_train = npzfile['Y_cadiz_train']
X_val = npzfile['X_val']
Y_val = npzfile['Y_cadiz_val']
X_test = npzfile['X_test']
Y_test = npzfile['Y_cadiz_test']

X_train = np.concatenate((X_train,X_val), axis=0)
Y_train = np.concatenate((Y_train,Y_val), axis=0)

ntrain = X_train.shape[0]
ntest = X_test.shape[0]

nclases_train = np.sum(Y_train, axis=0)
nclases_test = np.sum(Y_test, axis=0)
print('Classes train: [', nclases_train[0], ',', nclases_train[1], ',', nclases_train[2], ',', nclases_train[3], ']')
print('Classes test: [', nclases_test[0], ',', nclases_test[1], ',', nclases_test[2], ',', nclases_test[3], ']')

ok_train = np.argmax(Y_train, axis=1)
ok_test = np.argmax(Y_test, axis=1)

# Apply random forest
print('\nApplying Random Forest')
tini = time.time()
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, ok_train)
tfin = time.time()
print('Runtime: %.1f' %(tfin-tini))

pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)
naciertos_train = np.sum(pred_train==ok_train)
naciertos_test = np.sum(pred_test==ok_test)

# Confusion matrices
prec_train = np.zeros(4)
rec_train = np.zeros(4)
prec_test = np.zeros(4)
rec_test = np.zeros(4)

cm_train = confusion_matrix(ok_train, pred_train)
print('Confusion matrix train:')
print(cm_train)
for i in range(4):
    prec_train[i] = 100.0*cm_train[i,i]/np.sum(cm_train[:,i])
    rec_train[i] = 100.0*cm_train[i,i]/np.sum(cm_train[i,:])
print('Precision: %.2f, %.2f, %.2f, %.2f' % (prec_train[0], prec_train[1], prec_train[2], prec_train[3]))
print('Recall:    %.2f, %.2f, %.2f, %.2f' % (rec_train[0], rec_train[1], rec_train[2], rec_train[3]))

cm_test = confusion_matrix(ok_test, pred_test)
print('\nConfusion matrix test:')
print(cm_test)
for i in range(4):
    prec_test[i] = 100.0*cm_test[i,i]/np.sum(cm_test[:,i])
    rec_test[i] = 100.0*cm_test[i,i]/np.sum(cm_test[i,:])
print('Precision: %.2f, %.2f, %.2f, %.2f (mean: %.2f)' % (prec_test[0], prec_test[1], prec_test[2], prec_test[3], np.mean(prec_test)))
print('Recall:    %.2f, %.2f, %.2f, %.2f (mean: %.2f)' % (rec_test[0], rec_test[1], rec_test[2], rec_test[3], np.mean(rec_test)))
u = 90.0

# Save the model only if the precision and recall values are greater than u in test
if ((prec_test[0]>u) and (prec_test[1]>u) and (prec_test[2]>u) and (prec_test[3]>u) and (rec_test[0]>u) and (rec_test[1]>u) and (rec_test[2]>u) and (rec_test[3]>u)):
    print('Copying model in directory', flush=True)
    char_set = string.ascii_lowercase + string.digits
    nombre_dir = ''.join(random.sample(char_set, 6))
    os.mkdir(nombre_dir)
    joblib.dump(rf, nombre_dir+"/rf.gz")

