# -*- coding: utf-8 -*-
"""
    LSTM Performs an LSTM machine learning on the given data
    "

    Created by : newbi
    Created on Sun Dec  6 18:22:13 2020
    Modified on : 07.12.2020
    Based on : ML class at EPFL 
    Info : 
    
    UPDATE : 
    1. 
"""

import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


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
    Y_train = np_utils.to_categorical(dataY[:n_train])
    Y_test = np_utils.to_categorical(dataY[n_train:])
    
    return X_train, Y_train, X_test, Y_test
    

def create_LSTM_model():
    pass

def test_data():
    pass

    
if __name__ == "__main__":
    # if code ran standalone, perform LSTM
    
    # filenames
    in_file_path = './Data/data_150-4548_mem150.csv'
    out_file_path = './Data/cleaned_data.csv'
    # load the data
    df = pd.read_csv(in_file_path,index_col=0)
    a,b,c,d = prepare_data(df,10)