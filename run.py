# -*- coding: utf-8 -*-
"""
    RUN Performs an LSTM machine learning on the given data
    "

    Created by : newbi
    Created on Sat Dec  12 12:14:00 2020
    Modified on : 12.12.2020
    Based on : ML class at EPFL 
    Info : 
    
    UPDATE : 
    1. 
"""

import pandas as pd

from MK_run import mk_predict
from LSTM_run import lstm_predict
from RF_run import rf_predict
from TCN_run import tcn_predict

from Helpers.p_indicators import p_inds
from Helpers.performance_comparison import perf_comp

def run_models(model, history_window=10, hyperparam_opt=False, \
               results_file = './Data/Results.csv'):
    """ RUN_MODELS Runs the selected model with the given settings.
        Runs the selected model and then prints out the metrics and generates
        graphs.
    

    Parameters
    ----------
    model : String
        Which model to run. Either "MK", "LSTM", "RF" or "TCN".
    hyperparam_opt : Boolean
        Whether to perform a hyperparameter optimization or not.
    history_window : int
        How many days to consider in memory.

    Returns
    -------
    None.

    """
    
    print("Performing prediction with {} model.".format(model.upper()))
    if model.upper() == "MK":
        pass
        y_pred, y_true = mk_predict()
    elif model.upper() == "LSTM":
        pass
        y_pred, y_true = lstm_predict(hyperparam_opt, history_window)
    elif model.upper() == "RF":
        pass
        y_pred, y_true = rf_predict(history_window, hyperparam_opt)
    elif model.upper() == "TCN":
        y_pred, y_true = tcn_predict(hyperparam_opt, history_window)
    else:
        print("Unknown model chosen: ", model)
        
    # get the performance indicators
    print("Calculating performance indicators.")
    class_dict = p_inds(y_true, y_pred)
    
    # load previous indicators    
    df = pd.read_csv(results_file)
    # find which row to update
    row_idx = (df['Model'] == model.upper()) & \
                 (df['Memory'] == history_window)
    # update current row
    df.loc[row_idx,'Model'] = model.upper()
    df.loc[row_idx,'Memory'] = history_window
    df.loc[row_idx,'accuracy'] = class_dict['accuracy']
    df.loc[row_idx,'precision'] = class_dict['weighted avg']['precision']
    df.loc[row_idx,'f1'] = class_dict['weighted avg']['f1-score']
    df.loc[row_idx,'recall'] = class_dict['weighted avg']['recall']
  
    # save the data
    df.to_csv(results_file,index=False)
  
    # save the new figures
    perf_comp(df)


if __name__ == "__main__":
    # initialize variables
    model = ""
    hyperparam_opt = False
    history_window = 1
    
    # set list of valid models
    valid_models = ['MK', 'LSTM', 'RF','TCN']
    
    # find which model to use
    while model.upper() not in valid_models:
        model = input("Please choose a model:\n Choices are 'MK', 'LSTM', 'RF','TCN'\n")
    print("The {} model will be used.".format(model.upper()))
    
    # if not using Markov, ask the other parameters
    if model.upper() != "MK":
        hyperparam_opt = input("Would you like to perform hyperparameter tuning (1 for true/0 for false)?\n")
        history_window = input("How long would you like the history window to be (days)?\nMust be 2, 10, 50, 100 or 150 if no tuning is done.\n")
    
    # sanitize user input for hyperparam
    if hyperparam_opt == "1":
        print("Hyperparameter optimization will be performed.")
        hyperparam_opt = True
    else:
        print("Hyperparameter optimization will NOT be performed.")
        hyperparam_opt = False
        
    # sanitize user input for history window
    try:
        if int(history_window) >= 1 & int(history_window) <= 150:
            print("History window will be set to {} days.".format(history_window))
            history_window = int(history_window)
        else:
            print("Invalid history window chosen. History window will be set to 1 day.")
            history_window = 1
    except:
        print("Invalid history window chosen. History window will be set to 1 day.")
        history_window = 1

    # run the model
    run_models(model, history_window, hyperparam_opt)