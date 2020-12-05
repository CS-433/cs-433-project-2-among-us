# -*- coding: utf-8 -*-
"""
    RELABEL Relabels the state data so they are consistent over days
    "

    Created by : newbi
    Created on Sat Nov 28 11:01:21 2020
    Modified on : 28.11.2020
    Based on : ML class at EPFL 
    Info : 
    
    UPDATE : 
    1. 
"""

import pandas as pd
import numpy as np
from itertools import compress
from sklearn.metrics.cluster import adjusted_rand_score
import difflib

def find_similarity(df):
    """ FIND_SIMILARITY Returns the similarity of elements in a dataframe
        Finds the similarity between clustering in a dataframe with the
        cluster of the previous row.
    

    Parameters
    ----------
    df : Pandas dataframe (N x D)
        Dataframe of N datapoints with D features
        on which to check the similarity. Assumes it is a time series
        and therefore elements have to be shifted by one to properly compare

    Returns
    -------
    similarities : Numpy array (N - 1 x 1)
        Array of similarities between elements i and i+1.
        Contains N-1 elements because of the similarity compared relatively

    """

    similarities = np.zeros(len(df)-1) # initialize the similarity array
    prev_row_labels = df.iloc[0].tolist() # prepare previous row labels
    
    # loop through all (but zeroeth) elements of dataframe
    for i in range(1,len(df)):
        curr_row_labels = df.iloc[i].tolist() # find labels of the row as a list
        # get the similarity. shifted by one since it is a time series
        similarities[i-1] = adjusted_rand_score(curr_row_labels[1:-1], \
                                                prev_row_labels[0:-2])
        # assign previous row labels for next iteration
        prev_row_labels = curr_row_labels
    
    return similarities

def get_indices(df, index, state):
    elements = np.array(df.iloc[index] == state) # bool of elements with label
    indices = np.flatnonzero(elements * np.array(range(0,len(elements))))
    return indices

def compare(df, index):    
    iminus1_label_count = df.iloc[index-1].max() + 1
    i_label_count = df.iloc[index].max() + 1
    
    similitudes = np.zeros([iminus1_label_count,i_label_count])
    
    for j in range (0,iminus1_label_count):
        for k in range(0,i_label_count):
            sm = difflib.SequenceMatcher(None, \
                 get_indices(df,index-1,j)+1,get_indices(df,index,k))
            similitudes[j,k] = sm.ratio()
    return similitudes

def relabel_states(df, cutoff):
    # set highest index as number labels on first day
    highest_index = df.iloc[0].max() + 1
    # loop through all the days in the dataframe
    for i in range(1,len(df)):
        row_data = df.iloc[i]
        copied_data = df.iloc[i].copy() # copy to keep original labels
        i_label_count = row_data.max() + 1
        sims = compare(df,i)
        max_sims = sims.max(axis=0)
        best_fits_indices = sims.argmax(axis=0)
        # loop through all the states on that day
        for j in range(0,i_label_count):
            if max_sims[j] > cutoff:
                row_data[copied_data==j] = best_fits_indices[j]
            else:
                highest_index += 1
                row_data[copied_data==j] = highest_index
                

    
if __name__ == "__main__":
    in_file_path = './Data/data_150-4548_mem150.csv'
    out_file_path = './Data/cleaned_data.csv'
    df = pd.read_csv(in_file_path,index_col=0)
    df2 = df.iloc[0:29]
    relabel_states(df2,0.5)