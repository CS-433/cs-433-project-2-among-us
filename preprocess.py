# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:37:30 2020
Performs the preprocessing on the data

@author: newbi
"""
import pandas as pd

from Helpers.state_estimation import get_states
from Helpers.relabel import relabel_states

if __name__ == "__main__":
    # execute only if run as a script
    labelled_file_path = './Data/Labelled_data_150_4548_mem150.csv'
    preprocessed_file_path =  './Data/preprocessed.csv'
    
    # get the states for days 150 to 4548
    data = get_states(start_date = 150, end_date = 4548, memory = 150)
    # save to CSV
    data.to_csv(labelled_file_path,index=True)
    # read a CSV if the states were already obtained
    #data = pd.read_csv(labelled_file_path,index_col=0)
    # relabel data so it matches between days
    relabel_states(data,0,10)
    # save to csv
    data.to_csv(preprocessed_file_path)