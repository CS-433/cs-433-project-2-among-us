# -*- coding: utf-8 -*-
"""
Function RANDOM FOREST.
Created by : BenMobility
Created on : 22.11.2020
Modified on : 12.12.2020

UPDATE : 
    1. added pickle 
    2. added the title for the performance indicators, hyperparameter tuning.
    3. make the code as function itself and add a sys path to get a folder
    4. delete the rf_helpers and add train_test_split function here
"""
#usual imports
import pandas as pd
from pprint import pprint
import pickle
import numpy as np

# scikitlearn helpers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# split function
def train_test_split(data, n_train):
    """
    SPLIT TRAIN AND TEST sets from a univariate dataset

    Parameters
    ----------
    data : n x d dataframe pandas
        it contains the label and the features of the dataset.
    n_train : scalar
        tells how much rows you want to keep for the train set

    Returns
    -------
    TYPE
        train dataset from the first row to the scalar input.
    TYPE
        test dataset starts from the scalar input row until the end of the 
        original dataset.
    """
    return data[:n_train, :], data[n_train:, :]


#random forest prediction
def rf_predict(nbday, tuning = True):
    """
    Parameters
    ----------
    tuning : True / False, optional
        If tuning it is true, the function will go through the hyperparameter
        tuning with already selected parameter grid. If false, it will select
        the pickle with the provided number of days mem. The default is True.
    nbdays : integer
        Number of days memory between [2, 10, 50, 100, 150]

    Returns
    -------
    y_predicted : n x 1, integers
        It returns the predicted values of a 80/20 split from the prepro data
    """
    #check nbdays
    check = [2, 10, 50, 100, 150]
    if nbday not in check:
        print('\nWrong number of days!\n')
        print('Please choose between: 2, 10, 50, 100, 150')
        
    if nbday in check:
        #filename
        filename = 'Models\RF\RF_pickle{}'.format(nbday)
        
        #add one to nb of days to consider the first consider as the label
        nbdays = nbday + 1
        # load the dataset
        PATH = "Data\preprocessed.csv"
        data = pd.read_csv(PATH, header=0, index_col=0)
        data = data.to_numpy()
        data = data[:,:nbdays]
        
        #train/test split ratio
        ratio = 0.80
        n_train = int(len(data) * ratio)
        train, test = train_test_split(data, n_train)
        
        #call the classifier
        rf = RandomForestClassifier(random_state = 17)
        
        if tuning == True:
        
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 3)]
            # Number of features to consider at every split
            max_features = ['auto']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [8, 10, 12]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [3, 4, 5]
            # Method of selecting samples for training each tree
            bootstrap = [True]
            
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            print()
            print('Random grid for RF:\n')
            pprint(random_grid)
            
            # Random search of parameters, using 3 fold cross validation, 
            # search across 100 different combinations, and use all available cores
            rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                          n_iter = 100, scoring='f1_weighted', 
                                          cv = 3, verbose=2, random_state=17, n_jobs=-1,
                                          return_train_score=True)
            # data for the train
            train_features, train_labels = train[:, 1:], train[:, 0]
            
            # Fit the random search model
            rf_random.fit(train_features, train_labels)
            
            # Get the best params
            best_grid = rf_random.best_params_
            best_random = rf_random.best_estimator_
            print(best_grid)
        
            #pickle
            pickle_out = open(filename,"wb")
            pickle.dump(best_random, pickle_out)
            pickle_out.close()
            
        #open the pickle
        pickle_in = open(filename, "rb")
        best_random = pickle.load(pickle_in)
        
        # test features and labels
        # data for the train
        test_features, y_true = test[:, 1:], test[:, 0]
        y_predicted = best_random.predict(test_features)
    return y_predicted, y_true