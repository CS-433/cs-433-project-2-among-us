# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:16:45 2020

@author: PAHU95377
"""
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
import numpy as np

# scikitlearn helpers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#plotting and stats
from p_indicators import p_inds, con_matrix

#random forest helpers
from rf_helpers import train_test_split

#%% LOAD DATASET AND BASELINE RF
# load the dataset
PATH = "Data\cleaned_data_cutoff0_memory10_sparse_removed.csv"
CSVNAME = "cleaned_data_cutoff0_memory10_sparse_removed"
data = pd.read_csv(PATH, header=0, index_col=0)
data = data.to_numpy()

#train/test split ratio
ratio = 0.97
n_train = int(len(data) * ratio)
n_test = len(data)-n_train
train, test = train_test_split(data, n_train)

#%% HYPERPARAMETER TUNING RF
rf = RandomForestClassifier()

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

#%% RF
rf = RandomForestClassifier(random_state = 17)
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
#%%
# Get the best params
best_grid = rf_random.best_params_
best_random = rf_random.best_estimator_
print(best_grid)
#%% PICKLE 
filename = 'rf_cv_best_random'
pickle_out = open(filename,"wb")
pickle.dump(best_random, pickle_out)
pickle_out.close()

#%% EVALUATE
filename = 'rf_cv_best_random'
pickle_in = open(filename, "rb")
best_random = pickle.load(pickle_in)

# test features and labels
# data for the train
test_features, test_labels = test[:, 1:], test[:, 0]
test_predict = best_random.predict(test_features)

#%% PERFORMANCE
class_dict = p_inds(test_labels, test_predict,'rf_cv_step1') #add the name of the model
con_matrix(test_labels, test_predict, 'rf_cv_step1_matrix')

#%% Second round of grid search
#based on the random search best performance

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [30, 60],
    'max_features': [15,20],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [1200,1600]
}

# Create a base model
rf = RandomForestClassifier(random_state = 17)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1,\
                              scoring = 'f1_weighted',verbose = 2,\
                                  return_train_score=True)

# data for the train
train_features, train_labels = train[:, 1:], train[:, 0]

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)

grid_best_2 = grid_search.best_params_
best_grid_2 = grid_search.best_estimator_

print(best_grid_2)
#%% PICKLE 
filename = 'rf_cv_best_grid_2'
pickle_out = open(filename,"wb")
pickle.dump(best_grid_2, pickle_out)
pickle_out.close()

#%% EVALUATE
filename = 'rf_cv_best_grid_2'
pickle_in = open(filename, "rb")
best_grid_2 = pickle.load(pickle_in)

# test features and labels
# data for the train
test_predict_2 = best_grid_2.predict(test_features)

#%% PERFORMANCE
class_dict = p_inds(test_labels, test_predict,'rf_cv_step2') #add the name of the model
con_matrix(test_labels, test_predict, 'rf_cv_step2_matrix')

#%% Final search
param_grid = {
    'bootstrap': [True],
    'max_depth': [20,30, None],
    'max_features': [3],
    'min_samples_leaf': [3,4],
    'min_samples_split': [10],
    'n_estimators': [1400,1600]
}

# Create a base model
rf = RandomForestClassifier(random_state = 17)

# Instantiate the grid search model
grid_search_final = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                 cv = 3, n_jobs = -1, verbose = 2,\
                                     scoring = 'f1_weighted',\
                                         return_train_score=True)

grid_search_final.fit(train_features, train_labels)

grid_best_final = grid_search_final.best_params_
best_grid_final = grid_search_final.best_estimator_

test_predict_3 = best_grid_final.predict(test_features)

#%% PICKLE 
filename = 'rf_cv_best_grid_final'
pickle_out = open(filename,"wb")
pickle.dump(best_grid_2, pickle_out)
pickle_out.close()

#%% EVALUATE
filename = 'rf_cv_best_grid_final'
pickle_in = open(filename, "rb")
best_grid_final = pickle.load(pickle_in)

#%% PERFORMANCE
class_dict = p_inds(test_labels, test_predict,'rf_cv_final') #add the name of the model
con_matrix(test_labels, test_predict, 'rf_cv_final_matrix')

