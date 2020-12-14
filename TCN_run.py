#%%
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding
from keras.utils import np_utils
from kerastuner.tuners import RandomSearch
from tcn import TCN
import Helpers.p_indicators as p_ind
df = pd.read_csv('./Data/preprocessed.csv', index_col=0, parse_dates=True)
df_numpy = df.to_numpy()
N = len(df_numpy)
n_train = int(N* 0.8)
n_test = N - n_train
n_classes = len(np.unique(df_numpy))
dataY = df_numpy[:,0]
dataY = np_utils.to_categorical(dataY)
dataX = df_numpy[:,1:]
<<<<<<< Updated upstream
Y_train = np_utils.to_categorical(dataY[:n_train])
Y_test = np_utils.to_categorical(dataY[n_train:])
=======
Y_train = dataY[:n_train]
Y_test = dataY[n_train:]
>>>>>>> Stashed changes
memory=50
dataX[n_train:,:memory].shape
X_train = np.reshape(dataX[:n_train,:memory], (n_train, memory, 1))
X_test = np.reshape(dataX[n_train:,:memory], (n_test, memory,1))
X_train = X_train / float(n_classes)
X_test = X_test / float(n_classes)
i = Input(shape=(memory, 1))
<<<<<<< Updated upstream
#o = TCN()(i)
o = TCN(nb_filters = 64, kernel_size=6, dilations=[1,2,4,8,16,32,64])(i)
#m = Dropout(0.5)(i)
=======
o = TCN(nb_filters = 6, kernel_size=9, nb_stacks=3, dilations=[2 ** i for i in range(10)], padding='causal', dropout_rate=0.6)(i)
#o = Dropout(0.6)(o)
>>>>>>> Stashed changes
o = Dense(Y_train.shape[1], activation='softmax')(o)
model = Model(inputs=[i], outputs=[o])
model.summary()
model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30)
prediction = model.predict(X_test, verbose=0)
y_pred = np.argmax(prediction, axis=1)
Y_test_val = np.argmax(Y_test,axis=1)
p_ind.p_inds(Y_test_val,y_pred,'gggg')
#%%

def build_model(hp):
    i = Input(shape=(memory, 1))
    o = TCN(nb_filters = hp.Int('nb_filters', min_value=32, max_value=512, step=32), kernel_size=hp.Int('kernel', min_value=2, max_value=10, step=1), nb_stacks=2, dilations=[1,2,4,8,16,32], padding='causal')(i)
    #o = Dropout(0.5)(o)
    o = Dense(Y_train.shape[1], activation='softmax')(o)
    model = Model(inputs=[i], outputs=[o])
    model.summary()
#model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 'categorical_crossentropy', metrics=['accuracy'])
    model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
    return model

print('Train...')
<<<<<<< Updated upstream
model.fit(X_train, Y_train, epochs=30)
prediction = model.predict(X_test, verbose=0)
y_pred = np.argmax(prediction, axis=1)
Y_test_val = np.argmax(Y_test,axis=1)
p_ind.p_inds(Y_test_val,y_pred,'g')
=======

tuner = RandomSearch(build_model,objective='val_accuracy',max_trials=5,executions_per_trial=3,directory='C:/Users/loren/OneDrive/Documenti/my_dir')

tuner.search_space_summary()

tuner.search(X_train,Y_train,epochs=5,validation_data=(X_test, Y_test))
tuner.results_summary()
print('ciao')
best_model = tuner.get_best_models(num_models=2)
#best_model.fit(X_train, Y_train, epochs=30)
prediction = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(prediction, axis=1)
Y_test_val = np.argmax(Y_test,axis=1)
p_ind.p_inds(Y_test_val,y_pred,'g')
>>>>>>> Stashed changes
