import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import random
import string
import shutil
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.regularizers import l2

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

# Create the neural network
print('\nTraining')
landa = 1e-4
epochs = 3000
capas = 4
batch_size = 512
activacion = 'tanh'
inicializacion = 'glorot_uniform'
optimizador = tf.keras.optimizers.Adam()
funcion_coste = tf.keras.losses.CategoricalCrossentropy()
metrica = tf.keras.metrics.CategoricalCrossentropy()

# Model
model = Sequential()
model.add(Dense(units=100, input_shape=(18,), activation=activacion, kernel_regularizer=l2(landa), kernel_initializer=inicializacion))
for c in range(capas-1):
    model.add(Dense(units=100, activation=activacion, kernel_regularizer=l2(landa), kernel_initializer=inicializacion))
model.add(Dense(units=3, activation='softmax', kernel_regularizer=l2(landa), kernel_initializer=inicializacion))

model.summary()
model.compile(optimizer=optimizador, loss=funcion_coste, metrics=['accuracy',metrica])
es_callback = keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy', mode='min', patience=300, restore_best_weights=True)
lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_crossentropy', factor=0.01, patience=100, verbose=1, min_lr=1e-4, mode='min')
tini=time.time()
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val,Y_val), callbacks=[es_callback,lr_callback])
tfin=time.time()
print('Runtime train: %.1f' %(tfin-tini))

print('\nMax_epochs: %d, layers: %d, batch_size: %d, initialization: %s, activation: %s' % (epochs, capas, batch_size, inicializacion, activacion), flush=True)

char_set = string.ascii_lowercase + string.digits
nombre_dir = ''.join(random.sample(char_set, 6))
fich_modelo = 'modelo_'+nombre_dir+'.h5'
fich_loss = 'loss_'+nombre_dir+'.pkl'
model.save(fich_modelo)

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

# Save categorical_cross_entropy
loss = history.history['categorical_crossentropy']
val_loss = history.history['val_categorical_crossentropy']
f = open(fich_loss,'wb')
pickle.dump([loss,val_loss], f)
f.close()

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
print('Precision: %.2f, %.2f, %.2f (mean: %.2f)' % (prec_test[0], prec_test[1], prec_test[2], np.mean(prec_test)))
print('Recall:    %.2f, %.2f, %.2f (mean: %.2f)' % (rec_test[0], rec_test[1], rec_test[2], np.mean(rec_test)))

# Save the model only if the precision and recall values are greater than u in validation and test
u = 90.0
if ((prec_test[0]>u) and (prec_test[1]>u) and (prec_test[2]>u) and (rec_test[0]>u) and (rec_test[1]>u) and (rec_test[2]>u)):
    if ((prec_val[0]>u) and (prec_val[1]>u) and (prec_val[2]>u) and (rec_val[0]>u) and (rec_val[1]>u) and (rec_val[2]>u)):
        print('Copying model in directory', flush=True)
        char_set = string.ascii_lowercase + string.digits
        nombre_dir = ''.join(random.sample(char_set, 6))
        os.mkdir(nombre_dir)
        shutil.copy('sets.npz',nombre_dir)
        shutil.copy(fich_modelo,nombre_dir)
        shutil.copy(fich_loss,nombre_dir)

