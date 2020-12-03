# -*- coding: utf-8 -*-
"""
Random forest helpers

Created by : BenMobility
Created on : Thu Dec 03 10:03:49 2020
Modified on :

Info :     
    
UPDATE : 
1.     
"""
#Usual imports
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


def hypertuning_rf(train):
    """
    

    Parameters
    ----------
    train : n x d dataframe with d (label and features) of the supervised
    machine learning

    Returns
    -------
    prints out the plot of the hyperparameter tuning

    """
    trainX, trainY = train[:, 1:], train[:, 0]
    RANDOM_STATE = 17
    
    # # Generate a binary classification dataset.
    # X, y = make_classification(n_samples=500, n_features=25,
    #                            n_clusters_per_class=1, n_informative=15,
    #                            random_state=RANDOM_STATE)
    
    # NOTE: Setting the `warm_start` construction parameter to `True` disables
    # support for parallelized ensembles but is necessary for tracking the OOB
    # error trajectory during training.
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(warm_start=True, oob_score=True,
                                   max_features="sqrt",
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(warm_start=True, max_features='log2',
                                   oob_score=True,
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(warm_start=True, max_features=None,
                                   oob_score=True,
                                   random_state=RANDOM_STATE))
    ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    
    # Range of `n_estimators` values to explore.
    min_estimators = 10
    max_estimators = 200
    
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(trainX, trainY)
    
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))
    
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)
    
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()
    return

# TRAIN_TEST_SPLIT
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

# RANDOM FOREST FORECAST BASE MODEL
def rf_forecast(train, testX, best_grid):
    """
    function : RANDOM FOREST CLASSIFIER FORECAST
    fit an random forest model and make a one step prediction

    Parameters
    ----------
    train : n_train x d dataframe which includes the label and the features
        you should normally go through a test_train_split before
    testX : n_test x d-1 dataframe which only includes the features.
        you should normally extract the test label before putting here.
    best_grid : dict that contains all the best parameters for RFC,but you 
        need to do a grid search on a train_validate.
        
    Returns
    -------
    y_hat: the predicted label of n_test rows.

    """
	# transform list into array
    train = np.asarray(train)
	# split into input and output columns
    trainX, trainy = train[:, 1:], train[:, 0]
	# fit model
    model = RandomForestClassifier(**best_grid)
    model.fit(trainX, trainy)
	# make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# WALK-FORWARD VAL
def walk_forward_validation(data, n_train, best_grid):
    """
    function : Walk-forward validation for univariate data
    One step at a time, we predicted the next day label and then
    we include this prediction in the next training data for the next day
    and we walk forward until the last day we need to predict.

    Parameters
    ----------
    data : n x d dataframe of the label and the features of supervised 
    machine learning dataset.
    n_train : scalar
        a number that decides how many rows we want to keep as train set.
    best_grid : dict that contains all the best parameter for the RFC

    Returns
    -------
    error : scalar
        gives you back the total error (mean absolute error) of the model
        predictions with the test set. 
    y_true : n x 1 array
        the true label of the test set
    predictions : n x 1 array
        the predicted label with the model
    x_test : n x d-1
        the features of the test set

    """
    predictions = list()
	# split dataset
    train, test = train_test_split(data, n_train)
	# seed history with training dataset
    history = [x for x in train]
    # print the number of iteration
    print('The number of iteration will be:\n')
    print(len(test))
	# step over each time-step in the test set
    for i in range(len(test)):
		# split test row into input and output columns
        testX, testy = test[i, 1:], test[i, 0]
		# fit model on history and make a prediction
        yhat = rf_forecast(history, testX, best_grid)
		# store forecast in list of predictions
        predictions.append(yhat)
		# add actual observation to history for the next loop
        history.append(test[i])
		# summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
    error = mean_absolute_error(test[:, 0], predictions)
    return error, test[:, 0], predictions, test[:,1:]
