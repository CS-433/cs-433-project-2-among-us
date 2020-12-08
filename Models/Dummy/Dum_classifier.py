# -*- coding: utf-8 -*-
"""
Dummy machine learning classifier for the market states prediction
Created by : BenMobility
Created on Sat Dec  5 15:29:25 2020
Modified on : 

UPDATE : 

"""
#%% USUAL IMPORTS WITH RANDOM FOREST
import numpy as np
import pandas as pd

#plotting and stats
from matplotlib import pyplot
from p_indicators import p_inds, con_matrix

#random forest helpers
from rf_helpers import train_test_split

#random choices
from random import choices, seed

#%% LOAD DATASET AND BASELINE RF
# load the dataset
PATH = "Data\data_150-4548_mem150.csv"
data = pd.read_csv(PATH, header=0, index_col=0)
data = data.to_numpy()

#%%train/test split ratio
ratio = 0.80
n_train = int(len(data) * ratio)
n_test = len(data)-n_train
train, test = train_test_split(data, n_train)

#%% Weights of each label
a = train[:,0]
unique, counts = np.unique(a, return_counts=True)
figure= pyplot.figure()
weight = counts / sum(counts)
pyplot.bar(unique,weight)
pyplot.ylabel('Weights')
pyplot.xlabel('All states')
pyplot.xticks(ticks=range(0,14))
pyplot.xlim(-0.5,13.5)
pyplot.grid(which="major")
filename = 'weights'
pyplot.savefig('Figures\{}.pdf'.format(filename), dpi = 1080)
#%% Random choice with weights
seed(17)
yhat_rd = choices(unique,weight, k=(n_test))
y = test[:,0]

#%% Most frequent state 
most_frequent = np.argmax(weight)
yhat_mf = np.array([most_frequent]*n_test)
y = test[:,0]

#%% Markov first order
df = np.zeros((len(np.unique(data)), len(np.unique(data))))
for i in range(len(np.unique(train[:,0]))-1):
    t = train[train[:,0] ==i]
    unique, counts = np.unique(t[:,1], return_counts=True)
    weight = counts / sum(counts)
    for j in range(len(weight)-1):
        df[i,j] = weight[j]
yhat_mark = np.zeros(len(test))        
for i in range(len(test)):
    b = test[i,0]
    yhat_mark[i]=choices(np.unique(data), df[b,:])[0]

#%% PERFORMANCE
class_dict = p_inds(y, yhat_rd,"dummy_1") #add the name of the model
con_matrix(y, yhat_rd, 'dummy_1_splitratio_80')

class_dict = p_inds(y, yhat_mf,"dummy_2") #add the name of the model
con_matrix(y, yhat_mf, 'dummy_2_splitratio_80')

class_dict = p_inds(y, yhat_mark,"dummy_2") #add the name of the model
con_matrix(y, yhat_mark, 'dummy_2_splitratio_80')
