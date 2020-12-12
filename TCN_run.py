# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from tcn import TCN
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
memory=10
print(dataX.shape)
dataX[n_train:,:memory].shape
X_train = np.reshape(dataX[:n_train,:memory], (n_train, memory, 1))
X_test = np.reshape(dataX[n_train:,:memory], (n_test, memory,1))
X_train = X_train / float(n_classes)
X_test = X_test / float(n_classes)
i = Input(shape=(memory, 1))
m = TCN()(i)
m = Dense(Y_train.shape[1], activation='softmax')(m)

model = Model(inputs=[i], outputs=[m])
model.summary()
model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
print('Train...')
model.fit(X_train, Y_train, epochs=10)
predict = model.predict(X_train)
y_pred = np.argmax(predict, axis=1)
y_pred
y_unencoded = np.argmax(Y_train,axis=1)
plt.style.use("fivethirtyeight")
plt.figure(figsize = (15,7))
plt.plot(y_pred)
plt.plot(y_unencoded)
plt.title('Monthly Milk Production (in pounds)')
plt.legend(['predicted', 'actual'])
plt.xlabel("Months Counts")
plt.ylabel("Milk Production in Pounds")
plt.show()
