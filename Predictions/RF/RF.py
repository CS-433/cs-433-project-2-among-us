# -*- coding: utf-8 -*-
"""
Function RANDOM FOREST.
Created by : BenMobility
Created on : 22.11.2020
Modified on : 22.11.2020
Based on : ML class at EPFL - PROJECT 2
UPDATE : - 
"""
#%% USUAL IMPORTS WITH RANDOM FOREST
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from matplotlib import pyplot
from pindicators import p_inds

#%% TRAIN_TEST_SPLIT
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

#%% RANDOM FOREST FORECAST BASE MODEL
def rf_forecast(train, testX):
    """
    function : RANDOM FOREST CLASSIFIER FORECAST
    fit an random forest model and make a one step prediction

    Parameters
    ----------
    train : n_train x d dataframe which includes the label and the features
        you should normally go through a test_train_split before
    testX : n_test x d-1 dataframe which only includes the features.
        you should normally extract the test label before putting here.
        
    Returns
    -------
    y_hat: the predicted label of n_test rows.

    """
	# transform list into array
    train = np.asarray(train)
	# split into input and output columns
    trainX, trainy = train[:, 1:], train[:, 0]
	# fit model
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(trainX, trainy)
	# make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]

#%% WALK-FORWARD VAL
def walk_forward_validation(data, n_train):
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
	# step over each time-step in the test set
    for i in range(len(test)):
		# split test row into input and output columns
        testX, testy = test[i, 1:], test[i, 0]
		# fit model on history and make a prediction
        yhat = rf_forecast(history, testX)
		# store forecast in list of predictions
        predictions.append(yhat)
		# add actual observation to history for the next loop
        history.append(test[i])
		# summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
    error = mean_absolute_error(test[:, 0], predictions)
    return error, test[:, 0], predictions, test[:,1:]

#%% LOAD DATASET AND BASELINE RF
# load the dataset
PATH = "Data\data_150-4548_mem150.csv"
data = pd.read_csv(PATH, header=0, index_col=0)
data = data.to_numpy()
#train/test split ratio
ratio = 0.99
n_train = int(len(data) * ratio)
n_test = len(data)-n_train
# evaluate
mae, y, yhat, testX = walk_forward_validation(data, n_test)
print('MAE: %.3f' % mae)
# plot expected vs predicted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()

#%% Performance
p_inds(y, yhat)

#%% HYPERPARAMETER TUNING RF
rf = RandomForestClassifier(oob_score=True)

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

# Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

pprint(random_grid)

best_score = 0
for g in ParameterGrid(random_grid):
    rf.set_params(**g)
    rf.fit(testX,y)
    # save if best
    if rf.oob_score_ > best_score:
        best_score = rf.oob_score_
        best_grid = g

print( "OOB: %0.5f" % best_score )
print( "Grid:", best_grid)