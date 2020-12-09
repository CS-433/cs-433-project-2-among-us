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
        curr_row_labels = df.iloc[i].tolist() # find labels of the row as list
        # get the similarity. shifted by one since it is a time series
        similarities[i-1] = adjusted_rand_score(curr_row_labels[1:-1], \
                                                prev_row_labels[0:-2])
        # assign previous row labels for next iteration
        prev_row_labels = curr_row_labels
    
    return similarities


def get_members_indices(df, index, state):
    """ GET_MEMBERS_INDICES Gets the members in a database row with a state
        From the database df, get all of the members of the row index that
        has the value given by state. Returns a numpy array of the members

    Parameters
    ----------
    df : Pandas dataframe
        The dataframe to find members from.
    index : int
        The index of the row from which to find members.
    state : int
        The state of the members of interest from which to get the indices.

    Returns
    -------
    indices : Numpy array (N x 1)
        A numpy array containing a list of the elements that match the state.

    """
    
    elements = np.array(df.iloc[index] == state) # bool of elements with label
    # transform the boolean into a numpy array of the indices
    indices = np.flatnonzero(elements)
    return indices


def compare_days(df, index, offset=1, comp_labels=np.array([])):
    """ COMPARE_DAYS Compares the labels between two days
        Does a pairwise comparison of the labels between two days separated
        by an offset computes the similarity between all the pairs
    

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe on which to do the comparison.
    index : int
        Index of the day to take for comparison with the previous day.
    offset : int
        Number of days between both days to compare
    comp_labels : numpy array
        Labels to compare on day i with the previous labels.
        Defaults to all of the labels if none is given
        

    Returns
    -------
    similitudes : numpy array (N x M)
        Numpy array of the similitudes between the daily labels.
        Rows represent the labels from day 'index-1'
        Columns represent the labels from day 'index'

    """
    
    # get the labels for both days
    i_prev_labels = np.unique(df.iloc[index-offset])
    # no indices to check were given, check them all!
    if comp_labels.size == 0:
        comp_labels = np.unique(df.iloc[index])
    
    # initialize the similitude matrix
    similitudes = np.zeros([len(i_prev_labels),len(comp_labels)])
    
    # loop through the labels from the previous day
    for j, prev_label in enumerate(i_prev_labels):
        # loop through the labels to compare of the current day
        for k, curr_label in enumerate(comp_labels):
            # get the similitude for the day
            sm = difflib.SequenceMatcher(None, \
                 get_members_indices(df,index-offset,prev_label)+offset,\
                 get_members_indices(df,index,curr_label))
            # save the similitude ratio in the similitude matrix
            similitudes[j,k] = sm.ratio()
            
    return similitudes


def get_best_sims(df, index, window, i_labels=np.array([])):
    """ GET_BEST_SIMS Finds the best similitudes in the given window
        For a given day in the database df, find the best similitude in the
        given window. Returns the similitudes, the indices where this occured
        as well as the value of the label
    

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe on which to get the best similitudes.
    index : int
        Index of the row that will be compared
    window : int
        Number of days to take into consideration to compare with
    i_labels : numpy array
        Labels to compare on day i with the previous labels.
        Defaults to all of the labels if none is given

    Returns
    -------
    best_sims : Pandas dataframe.
        Dataframe containing the best similitudes and their information
            - index: the state of the row in the original dataframe, ordered
                    by unique() function
            - day : the day in which the best match was found
            - day_index : the index of the best match in the relative day,
                    ordered by unique() function
            - state_label : the label of the state that matches the best
            - val : the value of similitude between the matches

    """
        
    # no indices to check were given, check them all!
    if i_labels.size == 0:
        i_labels = np.unique(df.iloc[index])
    i_count = len(i_labels) # number of labels
    
    # setup the Pandas Dataframe
    best_sims = pd.DataFrame({'day' : [0] * i_count, \
                              'day_index' : [0] * i_count, \
                              'state_label': [0] * i_count,\
                              'val' : [0] * i_count})
    
    # loop through the previous days in the window
    for offset_days in range(1,window+1):
        # make sure we don't compare with the given day if negative
        # this is to ensure the first days that are smaller than the window
        # don't compare with negative days
        if index - offset_days >= 0:
            # similitudes with the previous day of reference
            sims = compare_days(df, index, offset_days, comp_labels=i_labels)
            max_sims = sims.max(axis=0) # find the most similar labels
            best_fits_indices = sims.argmax(axis=0) # index of best labels
            
            # get a boolean array to see if new similitudes are better
            new_sim_better = (max_sims > best_sims['val']).values
            
            # update the values that were better
            best_sims.loc[new_sim_better, 'day'] = index - offset_days
            best_sims.loc[new_sim_better, 'day_index'] = \
                best_fits_indices[new_sim_better]
            best_sims.loc[new_sim_better,'state_label'] = \
                np.unique(df.iloc[index - offset_days])\
                [best_fits_indices[new_sim_better]]
            best_sims.loc[new_sim_better,'val'] = max_sims[new_sim_better]
                
    return best_sims
    

def relabel_states(df, cutoff, window, sparse_cutoff = 0):
    """ RELABEL_STATES Relabels the states so they match within a dataframe
        Goes through an entire dataframe df and relabels the states
        to ensure that they match between the various rows.
        Labels are compared between the previous days in the window
        and matched to the most similar label from the previous days.
        Labels must be similar enough to be matched (above the cutoff).
        If they are not, they are given a new unique label.
        If a label doesn not have enough members, it is relabelled as a
        'random' label
    

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe to relabel.
    cutoff : float
        Similitude cutoff above which similitude must be to assign same label.
    window : int
        Number of days to take into consideration to compare with
    sparse_cutoff : int
        Minimum number of elements to keep a label

    Returns
    -------
    None.

    """
    
    # set total number of labels as the number labels on 0th day
    number_labels = len(np.unique(df.iloc[0]))
    
    remove_sparse_labels(df.iloc[0], sparse_cutoff)     
    
    # loop through all the days in the dataframe
    for i in range(1,len(df)):
        print(i)
        # remove today's sparse labels
        remove_sparse_labels(df.iloc[i], sparse_cutoff)        
        
        # get information about today's labels
        copied_data = df.iloc[i].copy() # copy to keep original labels
        i_labels = np.unique(copied_data) # labels today
        
        # find the best similitudes with the previous days in the window
        best_sims = get_best_sims(df, i, window)   
        
        # loop through all the labels on that day
        for k, label in enumerate(i_labels):
            # dont relabel the sparse labels
            if label != -1:
                # if there is no matching label, then it is a brand new unique
                # label that has appeared in position zero. it should have a
                # new label
                if best_sims.loc[k, 'val'] == 0:
                    # if not, give it a new unique label
                    df.iloc[i][copied_data==label] = number_labels
                    number_labels += 1
                # if the best matching label is above cutoff
                elif best_sims.loc[k, 'val'] >= cutoff:
                    # then assign today's label to yesterday's matching label
                    # find the new state to give
                    df.iloc[i][copied_data==label] = \
                        int(best_sims.loc[k, 'state_label'])
                else:
                    # if not, give it a new unique label
                    df.iloc[i][copied_data==label] = number_labels
                    number_labels += 1
                    
    # add the next day's label as the first column to act as the Y
    data = df['w(i-0)']
    data = np.roll(data,-1)
    df.insert(0,'true label: w(i+1)', data)
    
    # drop the last row, which doesn't know the next label
    df = df[:-1]
                    
                    
def relabel_states_forced_diag(df, cutoff, window):
    """ RELABEL_STATES Relabels the states so they match within a dataframe
        Goes through an entire dataframe df and relabels the states
        to ensure that they match between the various rows.
        Forces the diagonal, meaning only the new label on that row is compared
        with the the previous days in the window
        and matched to the most similar label from the previous days.
        Labels must be similar enough to be matched (above the cutoff).
        If they are not, they are given a new unique label.
    

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe to relabel.
    cutoff : float
        Similitude cutoff above which similitude must be to assign same label.
    window : int
        Number of days to take into consideration to compare with

    Returns
    -------
    None.

    """
       
    # set total number of labels as the number labels on 0th day
    number_labels = len(np.unique(df.iloc[0]))   
    
    # loop through all the days in the dataframe
    for i in range(1,len(df)):
        print(i)
        
        # find today's label since it is the only new value since last row
        new_label = np.array([df.iloc[i,0]])
        
        # best similitude of today's label with the previous days in the window
        best_sims = get_best_sims(df, i, window, i_labels = new_label)
        
        # if the best matching label is above cutoff
        if best_sims.loc[0, 'val'] >= cutoff:
            # then assign today's label to yesterday's matching label
            # find the new state to give
            df.iloc[i,0] = \
                int(best_sims.loc[0, 'state_label'])
        else:
            # if not, give it a new unique label
            df.iloc[i,0] = number_labels
            number_labels += 1
        # change the rest of the row to match the previous row (force diag)
        # note: doesn't work, as rows can no longer find the closest match
        # the following line should be commented
        df.iloc[i,1:] = np.array(df.iloc[i-1,:-1])
                    
    # add the next day's label as the first column to act as the Y
    data = df['w(i-0)']
    data = np.roll(data,-1)
    df.insert(0,'true label: w(i+1)', data)
    
    # drop the last row, which doesn't know the next label
    df = df[:-1]
    
                
def remove_sparse_labels(df, qty_cutoff):
    """ REMOVE_SPARSE_LABELS Remove labels that don't appear often
        Remove the labels that don't appear often and reorder and renumber
        the label numbers accordingly.
    

    Parameters
    ----------
    df : Pandas dataframe or series
        Dataframe to remove sparse labels from.
    cutoff : int
        Minimum number of elements to be kept.

    Returns
    -------
    None.

    """
    
    # get counts of unique values
    values, count = np.unique(df, return_counts=True)
    
    # find labels that are too rare
    to_replace = values[count < qty_cutoff]
    indices_to_replace = df.isin(to_replace)
    
    # replace values
    df[indices_to_replace] = -1
    
    
def reorder_labels(df, qty_cutoff):
    """ REORDER_LABELS Removes sparse labels and reorders to fill holes left
        Goes through a dataframe and removes the sparse labels
        Then goes through all the holes that were left by removing sparse
        labels and reorganizes the array to fill the holes
    

    Parameters
    ----------
    df : Pandas dataframe
        Dataframe to reorder.
    qty_cutoff : int
        Minimum number of elements to keep the label.

    Returns
    -------
    None.

    """
    remove_sparse_labels(df, qty_cutoff)
    
    # get the unique labels left
    labels = np.unique(df)
    
    # lookup dictionary value -> i where values are sorted in order=
    mapping = {}
    for i in range(0,len(labels)):
        mapping[labels[i]] = i
        
    # replace the labels
    df.replace(mapping, inplace=True)
    

if __name__ == "__main__":
    # if code ran in a standalone version, relabel the given file
    
    # filenames
    in_file_path = './Data/data_150-4548_mem150_no_true.csv'
    out_file_path = './Data/cleaned_data_cutoff0_memory10.csv'
    # load the data
    data = pd.read_csv(in_file_path,index_col=0)
    # to do only part of the data, as it is quite long to relabel everything
    #data2 = data.iloc[:500].copy()
    # relabel the data
    relabel_states(data,0,10)
    #0.15,25 and 9 is stable
    # save to CSV
    #data.to_csv(out_file_path,index=True)
    #a = get_best_sims(data,12,15)
    
    # for i in range(0,len(data2)):
    #     un, ct = np.unique(data2.iloc[i],return_counts=True)
    #     count = ct[un[un==data2.iloc[i,0]]]
    #     if count == 1:
    #         print(i)
    #         #22,189,1224,2172,2197,2219,2320,2765,3143,3249,3547,3838,3953,4225
    #         #22,47,69,170