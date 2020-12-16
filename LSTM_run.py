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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from kerastuner import Hyperband, HyperModel


class MyHyperModel(HyperModel):
    """ MYHYPERMODEL Custom Hypermodel with vector shapes
        A custom hypermodel that includes the shapes of the vectors in its
        definition to create the layers with the right sizes.
    """

    def __init__(self, X_timesteps, X_features, Y_shape):
        """ __INIT__ Creates the hypermodel
        

        Parameters
        ----------
        X_timesteps : int
            Number of timesteps in the X dataset given.
        X_features : int
            Number of features in the X dataset given.
        Y_shape : int
            dimensions of the Y vector. Corresponds to number of classes

        Returns
        -------
        None.

        """
        self.X_timesteps = X_timesteps
        self.X_features = X_features
        self.Y_shape = Y_shape

    def build(self, hp):
        """ BUILD Build the custom HyperModel
        

        Parameters
        ----------
        hp : Hyperparameter list
            A list containing the hyperparameters to try.

        Returns
        -------
        model : HyperModel
            The custom hypermodel.

        """
        
        # select the parameters to tune
        unit_num = hp.Int('units',min_value=64, max_value=256, step=16)
        learn_rate = hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])
        act = hp.Choice('activation',
                               values=['relu', 'sigmoid', 'tanh'])
        num_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        
        # build the model
        model = Sequential()
        # input layer
        model.add(LSTM(units=unit_num, activation=act, \
                           input_shape=(self.X_timesteps, self.X_features),\
                        return_sequences=True))
        model.add(Dropout(0.2))
        # build the first hidden layers
        for i in range(0, num_layers):
            # determine if on the last layer or not
            # return sequences accordingly
            if i == num_layers - 1:
                ret_seq = False
            else:
                ret_seq = True
            model.add(LSTM(unit_num, activation=act,return_sequences=ret_seq))
            model.add(Dropout(0.2))
        
        # build the dense layer
        model.add(Dense(self.Y_shape, activation='softmax'))
    
        # compile the model
        model.compile(loss='categorical_crossentropy', \
                      optimizer=Adam(learn_rate),
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
    """ LSTM_PREDICT Performs an LSTM prediction on the given data
        Performs an LSTM prediction with or without hyperparameter optimization
        Returns the predictions and the test data that was used
    

    Parameters
    ----------
    hyperparam_opt : boolean
        Whether or not to perform hyperparameter optimization.
    history_window : int
        The number of days to take in memory for the prediction.

    Returns
    -------
    y_pred : numpy array
        The predicted labels of the test data.
    Y_test_val : numpy array
        The true labels of the test data..

    """
    
    # define input data path
    in_file_path = './Data/preprocessed.csv'
    # define the model path
    filepath="./Models/LSTM/model-lstm-{}mem.hdf5".format(history_window)
    
    # load the data
    df = pd.read_csv(in_file_path,index_col=0)
    # shape the data properly
    X_train, Y_train, X_test, Y_test = prepare_data(df, history_window)
     
    if hyperparam_opt:
        # perform hyperparam optimization
        
        hyp_epoch_num = 25
        final_epoch_num = 50
        
        # define the LSTM model
        hypermodel = MyHyperModel(X_train.shape[1], X_train.shape[2], \
                                  Y_train.shape[1])
        
        # Bayesian optimizer. not used anymore
        # tuner = BayesianOptimization(
        #     hypermodel,
        #     objective='val_acc',
        #     max_trials=1,
        #     executions_per_trial=1,
        #     directory='./Models/LSTM/',
        #     project_name='tuning_{}mem'.format(history_window),
        #     overwrite=True)
        
        # define the optimizer
        tuner = Hyperband(hypermodel,
                          objective='val_acc',
                          max_epochs=hyp_epoch_num,
                          factor=3,
                          hyperband_iterations=2,
                          executions_per_trial=2,
                          directory='./Models/LSTM/',
                          project_name='tuning_{}mem'.format(history_window),
                          overwrite=True)
        # perform the hyperparameter optimization
        tuner.search(X_train, Y_train,
             epochs=hyp_epoch_num,
             validation_data=(X_test, Y_test))
        
        # Get the optimal hyperparameters
        best_hps = \
            tuner.get_best_hyperparameters(num_trials = 1)[0]
        
        # build the best model
        best_model = hypermodel.build(best_hps)
        # lazy way to get the best model. removed since above is better
        #best_model = tuner.get_best_models(num_models=1)[0]
                
        # fit data to model
        best_model.fit(X_train, Y_train, epochs=final_epoch_num, batch_size=64)
        
        # print the best hyperparameters. done here after the fitting
        # so it remains on the console and doesn't get lost
        print(f"""
        The hyperparameter search is complete.\n
        Unit number: {best_hps.get('units')}\n
        Learn rate: {best_hps.get('learning_rate')}\n
        Activation function: {best_hps.get('activation')}\n
        Number of layers: {best_hps.get('num_layers')}\n
        """)
        
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
    lstm_predict(True, 3)
    
