# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding
from keras.utils import np_utils
from tcn import TCN
import Helpers.p_indicators as p_ind
df = pd.read_csv('./Data/preprocessed.csv', index_col=0, parse_dates=True)
df_numpy = df.to_numpy()
N = len(df_numpy)
n_train = int(N* 0.8)
n_test = N - n_train
n_classes = len(np.unique(df_numpy))
dataY = df_numpy[:,0]
dataX = df_numpy[:,1:]
Y_train = np_utils.to_categorical(dataY[:n_train])
Y_test = np_utils.to_categorical(dataY[n_train:])
memory=50
dataX[n_train:,:memory].shape
X_train = np.reshape(dataX[:n_train,:memory], (n_train, memory, 1))
X_test = np.reshape(dataX[n_train:,:memory], (n_test, memory,1))
X_train = X_train / float(n_classes)
X_test = X_test / float(n_classes)
i = Input(shape=(memory, 1))
#o = TCN()(i)
o = TCN(nb_filters = 64, kernel_size=6, dilations=[1,2,4,8,16,32,64])(i)
#m = Dropout(0.5)(i)
o = Dense(Y_train.shape[1], activation='softmax')(o)
model = Model(inputs=[i], outputs=[o])
model.summary()
model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
print('Train...')
model.fit(X_train, Y_train, epochs=30)
prediction = model.predict(X_test, verbose=0)
y_pred = np.argmax(prediction, axis=1)
Y_test_val = np.argmax(Y_test,axis=1)
p_ind.p_inds(Y_test_val,y_pred,'g')
