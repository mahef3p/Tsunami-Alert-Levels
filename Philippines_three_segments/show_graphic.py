import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import string
import shutil
import pickle
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def autolabel(rects):
    """Add a label in each bar showing the value"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -15),  # -15 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Read data sets
print('Reading X_train, Y_train, X_val, Y_val, X_test, Y_test')
npzfile = np.load('sets.npz')
X_train = npzfile['X_train']
Y_train = npzfile['Y_train']
X_val = npzfile['X_val']
Y_val = npzfile['Y_val']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']

ntrain = X_train.shape[0]
nval = X_val.shape[0]
ntest = X_test.shape[0]

nclases_train = np.sum(Y_train, axis=0)
nclases_val = np.sum(Y_val, axis=0)
nclases_test = np.sum(Y_test, axis=0)
print('Classes train: [', nclases_train[0], ',', nclases_train[1], ',', nclases_train[2], ']')
print('Classes val: [', nclases_val[0], ',', nclases_val[1], ',', nclases_val[2], ']')
print('Classes test: [', nclases_test[0], ',', nclases_test[1], ',', nclases_test[2], ']')

model = keras.models.load_model('modelo.h5')
model.summary()

pred_train = np.argmax(model.predict(X_train), axis=1)
pred_val = np.argmax(model.predict(X_val), axis=1)
pred_test = np.argmax(model.predict(X_test), axis=1)
ok_train = np.argmax(Y_train, axis=1)
ok_val = np.argmax(Y_val, axis=1)
ok_test = np.argmax(Y_test, axis=1)
naciertos_train = np.sum(pred_train==ok_train)
naciertos_val = np.sum(pred_val==ok_val)
naciertos_test = np.sum(pred_test==ok_test)
porcentaje_aciertos_train = naciertos_train*100.0/ntrain
porcentaje_aciertos_val = naciertos_val*100.0/nval
porcentaje_aciertos_test = naciertos_test*100.0/ntest

# train_clase[i] contains the accuracy (0-100) for the i-th class in the train set
# The same for val_clase and test_clase
train_clase = np.zeros(3)
val_clase = np.zeros(3)
test_clase = np.zeros(3)
for i in range(3):
    # We only consider the rows of ok_train with class i, and compare them with pred_train
    filas = (ok_train == i)
    nfilas = np.sum(filas)
    naciertos = np.sum(pred_train[filas] == ok_train[filas])
    train_clase[i] = naciertos*100.0/nfilas

for i in range(3):
    # We only consider the rows of ok_val with class i, and compare them with pred_val
    filas = (ok_val == i)
    nfilas = np.sum(filas)
    naciertos = np.sum(pred_val[filas] == ok_val[filas])
    val_clase[i] = naciertos*100.0/nfilas

for i in range(3):
    # We only consider the rows of ok_test with class i, and compare them with pred_test
    filas = (ok_test == i)
    nfilas = np.sum(filas)
    naciertos = np.sum(pred_test[filas] == ok_test[filas])
    test_clase[i] = naciertos*100.0/nfilas

# Confusion matrices
from sklearn.metrics import confusion_matrix
prec_train = np.zeros(3)
rec_train = np.zeros(3)
prec_val = np.zeros(3)
rec_val = np.zeros(3)
prec_test = np.zeros(3)
rec_test = np.zeros(3)

cm_train = confusion_matrix(ok_train, pred_train)
print('Confusion matrix train:')
print(cm_train)
for i in range(3):
    prec_train[i] = 100.0*cm_train[i,i]/np.sum(cm_train[:,i])
    rec_train[i] = 100.0*cm_train[i,i]/np.sum(cm_train[i,:])
print('Precision: %.2f, %.2f, %.2f' % (prec_train[0], prec_train[1], prec_train[2]))
print('Recall:    %.2f, %.2f, %.2f' % (rec_train[0], rec_train[1], rec_train[2]))

cm_val = confusion_matrix(ok_val, pred_val)
print('\nConfusion matrix val:')
print(cm_val)
for i in range(3):
    prec_val[i] = 100.0*cm_val[i,i]/np.sum(cm_val[:,i])
    rec_val[i] = 100.0*cm_val[i,i]/np.sum(cm_val[i,:])
print('Precision: %.2f, %.2f, %.2f' % (prec_val[0], prec_val[1], prec_val[2]))
print('Recall:    %.2f, %.2f, %.2f' % (rec_val[0], rec_val[1], rec_val[2]))

cm_test = confusion_matrix(ok_test, pred_test)
print('\nConfusion matrix test:')
print(cm_test)
for i in range(3):
    prec_test[i] = 100.0*cm_test[i,i]/np.sum(cm_test[:,i])
    rec_test[i] = 100.0*cm_test[i,i]/np.sum(cm_test[i,:])
print('Precision: %.2f, %.2f, %.2f' % (prec_test[0], prec_test[1], prec_test[2]))
print('Recall:    %.2f, %.2f, %.2f' % (rec_test[0], rec_test[1], rec_test[2]))

# Show accuracy per class
labels = ['Yellow', 'Orange', 'Red']
ancho_barra = 2.0
ancho_linea = 0.5
x = np.arange(0, 15.5, 7.5)

fig, ax = plt.subplots()
rect1 = ax.bar(x - ancho_barra, train_clase, width=ancho_barra, linewidth=ancho_linea, edgecolor='black', color='powderblue', label='Train set')
rect2 = ax.bar(x, val_clase, width=ancho_barra, linewidth=ancho_linea, edgecolor='black', color='navajowhite', label='Validation set')
rect3 = ax.bar(x + ancho_barra, test_clase, width=ancho_barra, linewidth=ancho_linea, edgecolor='black', color='palegreen', label='Test set')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([0,100])
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.text(0.44, -0.11, 'Alert level', transform=ax.transAxes)
ax.legend(loc='upper left', edgecolor='black', bbox_to_anchor=(0.15,1.15), ncol=3)
autolabel(rect1)
autolabel(rect2)
autolabel(rect3)
fig.set_size_inches(16.05/2.54, 12.2/2.54)
fig.tight_layout()
plt.show()

