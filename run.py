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

from DM_run import dm_predict
#from LSTM_run import lstm_predict
from RF_run import rf_predict
#from TCN_run import tcn_predict

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
        Which model to run. Either "Dummy", "LSTM", "RF" or "TCN".
    hyperparam_opt : Boolean
        Whether to perform a hyperparameter optimization or not.
    history_window : int
        How many days to.

    Returns
    -------
    None.

    """
    
    print("Performing prediction with {} model.".format(model))
    if model.lower() == "dummy":
        pass
        y_pred, y_true = dm_predict()
    elif model.lower() == "lstm":
        pass
        #y_pred, y_true = lstm_predict(hyperparam_opt, history_window)
    elif model.lower() == "rf":
        pass
        y_pred, y_true = rf_predict(history_window, hyperparam_opt)
    elif model.lower() == "tcn":
        pass
        #y_pred, y_true = tcn_predict(hyperparam_opt, history_windowR)
    else:
        print("Unknown model chosen: ", model)
        
    # get the performance indicators
    print("Calculating performance indicators.")
    class_dict = p_inds(y_true, y_pred)
    
    # load previous indicators    
    df = pd.read_csv(results_file)
    # get the current row of interest
    row = df.loc[(df['Model'] == model.lower()) & \
                 (df['Memory'] == history_window)]
    # change the new data
    row['accuracy'] = class_dict['accuracy']
    row['precision'] = class_dict['weighted avg']['precision']
    row['f1'] = class_dict['weighted avg']['f1-score']
    row['recall'] = class_dict['weighted avg']['recall']
    # save the new figures
    perf_comp(df)
    # save the data
    df.to_csv(results_file,index=False)


if __name__ == "__main__":
    # initialize variables
    model = ""
    hyperparam_opt = False
    history_window = 1
    
    # set list of valid models
    valid_models = ['dummy', 'lstm', 'rf','tcn']
    
    # find which model to use
    while model.lower() not in valid_models:
        model = input("Please choose a model:\n Choices are 'Dummy', 'LSTM', 'RF','TCN'\n")
    print("The {} model will be used.".format(model))
    
    # if not using dummy, ask the other parameters
    if model.lower() != "dummy":
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