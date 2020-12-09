# -*- coding: utf-8 -*-
"""
Function RANDOM FOREST.
Created by : BenMobility
Created on : 22.11.2020
Modified on : 5.12.2020

UPDATE : 
    1. added pickle 
    2. added the title for the performance indicators, hyperparameter tuning.
"""
#%% USUAL IMPORTS WITH RANDOM FOREST
import pandas as pd
from pprint import pprint
import pickle

# scikitlearn helpers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

#plotting and stats
from matplotlib import pyplot
from p_indicators import p_inds, con_matrix

#random forest helpers
from rf_helpers import hypertuning_rf,train_test_split,walk_forward_validation


#%% LOAD DATASET AND BASELINE RF
# load the dataset
PATH = "Data\cleaned_data_03_50_9.csv"
data = pd.read_csv(PATH, header=0, index_col=0)
data = data.to_numpy()
#train/test split ratio
ratio = 0.80
n_train = int(len(data) * ratio)
n_test = len(data)-n_train
train, test = train_test_split(data, n_train)

#%% HYPERPARAMETER TUNING RF
rf = RandomForestClassifier(oob_score=True)

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

# Randomized Search
# Number of trees in random forest
n_estimators = [1200] #[int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]# num = 12
# Number of features to consider at every split
max_features = ['sqrt'] #['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [30,50] #num = 6 [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [100,150,200] #[2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [10,20] #[1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }

pprint(random_grid)

# data for the train
trainX, trainY = train[:, 1:], train[:, 0]
best_score = 0
i = 0
print('number of loop', len(ParameterGrid(random_grid)))
for g in ParameterGrid(random_grid):
    rf.set_params(**g)
    rf.fit(trainX,trainY)
    i += 1
    print('loop', i)
    # save if best
    if rf.oob_score_ > best_score:
        best_score = rf.oob_score_
        best_grid = g

print( "OOB: %0.5f" % best_score )
print( "Best grid:", best_grid)
#%% PICKLE 
filename = "rf_best_grid_03_50_9"
pickle_out = open(filename,"wb")
pickle.dump(best_grid, pickle_out)
pickle_out.close()

#%% EVALUATE
filename = "rf_best_grid_03_50_9"
pickle_in = open(filename, "rb")
best_grid = pickle.load(pickle_in)

mae, y, yhat, testX = walk_forward_validation(data, n_train, best_grid)
print('MAE: %.3f' % mae)
# plot expected vs predicted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()

#%% PERFORMANCE
class_dict = p_inds(y, yhat,"RF_03_50_9") #add the name of the model
con_matrix(y, yhat, 'RF_03_50_9_matrix')

#%% Plotting Hyperparameter
hypertuning_rf(train, "hyper_RF_03_50_9") #add the name of the model