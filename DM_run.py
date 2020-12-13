# -*- coding: utf-8 -*-
"""
Dummy machine learning classifier for the market states prediction
Created by : BenMobility
Created on Sat Dec  12 15:29:25 2020
Modified on : 

UPDATE : 
    1- only take the markov dummy classifier
    2- make it a function out of it

"""
#usual imports
import numpy as np
import pandas as pd

#random choices
from random import choices

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

#dummy prediction
def dm_predict():
    """
    Parameters
    ----------
    None

    Returns
    -------
    y_predicted : n x 1, integers
        It returns the predicted values of a 80/20 split from the prepro data
    y_true : n x 1, integers
        the true labels of the dataset
    """
  
    # load the dataset
    PATH = "Data\preprocessed.csv"
    data = pd.read_csv(PATH, header=0, index_col=0)
    data = data.to_numpy()
    
    #split
    ratio = 0.80
    n_train = int(len(data) * ratio)
    train, test = train_test_split(data, n_train)

  
    # Markov first order
    #initialize dataframe
    df = np.zeros((len(np.unique(data)), len(np.unique(data))))
    
    #compute the weight for each unique label in the dataset
    for i in range(len(np.unique(train[:,0]))-1):
        t = train[train[:,0] ==i]
        unique, counts = np.unique(t[:,1], return_counts=True)
        weight = counts / sum(counts)
        
        # save the weights in df
        for j in range(len(weight)-1):
            df[i,j] = weight[j]
            
    #initialize the predidicted labels array
    yhat_mark = np.zeros(len(test))        
    
    # predicting the next label with the previous on the test dataset
    # with the random.choices library and their weights based on markov
    # first order
    for i in range(len(test)):
        b = test[i,0]
        yhat_mark[i]=choices(np.unique(data), df[b,:])[0]
        
    #get the true labels only
    y_true = test[:,0]
        
    return yhat_mark, y_true


