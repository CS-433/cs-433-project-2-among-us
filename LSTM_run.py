# -*- coding: utf-8 -*-
"""
    LSTM Performs an LSTM machine learning on the given data
    "

    Created by : newbi
    Created on Sun Dec  6 18:22:13 2020
    Modified on : 12.12.2020
    Based on : ML class at EPFL 
    Info : 
    
    UPDATE : 
    1. 
"""

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from Helpers.p_indicators import p_inds
from kerastuner import BayesianOptimization
from kerastuner import HyperModel


class MyHyperModel(HyperModel):

    def __init__(self, X_shape1, X_shape2, Y_shape):
        self.X_shape1 = X_shape1
        self.X_shape2 = X_shape2
        self.Y_shape = Y_shape

    def build(self, hp):
        unit_num = hp.Int('units',min_value=8, max_value=64, step=8)
        
        model = Sequential()
        model.add(LSTM(units=unit_num, activation='relu', \
                           input_shape=(self.X_shape1, self.X_shape2),\
                        return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(unit_num, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.Y_shape, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy', \
                      optimizer=Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),
                      metrics=['acc'])
        return model  


def prepare_data(df, memory, valid_ratio=0.8, form='timestep'):
    """ PREPARE_DATA Puts the data in a format ready for keras LSTM
        Prepares the data found in the database df into a format ready for
        keras LSTM using the last 'memory' days as a moving window.
    

    Parameters
    ----------
    df : Pandas dataframe (N x (D+1))
        Dataframe to prepare data with N datapoints each having D+1 features
        The D + 1 features represent the last D day states and the day's state
    memory : int
        Number of days to take for the moving window
    valid_ratio : float
        The fraction of data to take for the training set
        Default is 0.8
    form : string, either 'timestep' or 'feature'
        The format to give the X array. Either the last 'memory' days are
        taken as a feature or as a timestep.
        Default is'timestep'

    Returns
    -------
    X_train : Numpy array (N x (memory * valid_ratio) x 1) or
                          (N x 1 x (memory * valid_ratio))
        An array representing the features/timesteps for the training data
    Y_train : Numpy array (N x 1)
        A numpy array representing the true labels for the training data
    X_test : Numpy array (N x (memory * (1 - valid_ratio)) x 1) or
                          (N x 1 x (memory * (1 - valid_ratio)))
        An array representing the features/timesteps for the testing data
    Y_test : Numpy array (N x 1)
        A numpy array representing the true labels for the testing data

    """
    
    df_numpy = df.to_numpy() # put the dataframe in numpy
    
    # find quantities for splitting
    N = len(df_numpy)
    n_train = int(N * valid_ratio)
    n_test = N - n_train
    n_classes = len(np.unique(df_numpy))
    
    # get X and Y data
    dataX = df_numpy[:,1:]
    dataY = df_numpy[:,0]
        
    # reshape X to be [samples, time steps, features]
    if form == 'feature':
        X_train = np.reshape(dataX[:n_train,:memory], (n_train, 1, memory))
        X_test = np.reshape(dataX[n_train:n_test,:memory], (n_test, 1, memory))
    else: # default to timestep, even if form was wrongly defined
        X_train = np.reshape(dataX[:n_train,:memory], (n_train, memory, 1))
        X_test = np.reshape(dataX[n_train:,:memory], (n_test, memory, 1))
        
    # normalize
    X_train = X_train / float(n_classes)
    X_test = X_test / float(n_classes)
    
    # one hot encode the output variable
    Y_encoded = np_utils.to_categorical(dataY)
    Y_train = Y_encoded[:n_train]
    Y_test = Y_encoded[n_train:]
    
    return X_train, Y_train, X_test, Y_test

    
def lstm_predict(hyperparam_opt, history_window):
    # filename
    in_file_path = './Data/preprocessed.csv'
    # load the data
    df = pd.read_csv(in_file_path,index_col=0)
    X_train, Y_train, X_test, Y_test = prepare_data(df, history_window)
    
    # define the model path
    filepath="./Models/LSTM/model-lstm-{}mem.hdf5".format(history_window)
    
    if hyperparam_opt:
        # perform hyperparam optimization
        
        # define the LSTM model
        hypermodel = MyHyperModel(X_train.shape[1], X_train.shape[2], \
                                  Y_train.shape[1])
        # define the optimization model
        bayesian_opt_tuner = BayesianOptimization(
            hypermodel,
            objective='val_acc',
            max_trials=5,
            executions_per_trial=3,
            directory='./Models/LSTM/',
            project_name='tuning_{}mem'.format(history_window),
            overwrite=True)
        
        # perform the hyperparameter optimization
        bayesian_opt_tuner.search(X_train, Y_train,
             epochs=5,
             validation_data=(X_test, Y_test))
        
        # get the best model
        best_model = bayesian_opt_tuner.get_best_models(num_models=1)[0]
                
        # fit data to model
        best_model.fit(X_train, Y_train, epochs=50, batch_size=64)
        
        # save the model
        best_model.save(filepath)
        
    else:
        # don't perform hyperparameter optimization, just load the best model
        best_model = load_model(filepath)
    
  
    # predict
    prediction = best_model.predict(X_test, verbose=0)
    
    # un-encode the y data
    y_pred = np.argmax(prediction,axis=1)
    Y_test_val = np.argmax(Y_test,axis=1)
    
    return y_pred, Y_test_val

    
if __name__ == "__main__":
    lstm_predict(True, 10)
    
