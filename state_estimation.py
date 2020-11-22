# -- coding: utf-8 --
"""
    STATE_ESTIMATION Estimates the states for the stocks
    Generates the states for the stocks over various days and saves a CSV
    "

    Created by : newbi
    Created on : 22.11.2020
    Modified on : 22.11.2020
    Based on : ML class at EPFL 
    Info : 
    
    UPDATE : 
    1. 
"""
from numpy import linalg as LA
import numpy as np
import pandas as pd
import networkx as nx
import community

def compute_C_minus_C0(lambdas,v,lambda_plus):
    N=len(lambdas)
    C_clean=np.zeros((N, N))
  
    # _s stands for _structure below. Note that range(N-1) means that we do not include the maximum eigenvalue
    for i in range(N-1):
        if lambdas[i]>lambda_plus:
            C_clean=C_clean + lambdas[i]*(np.outer(v[:,i],v[:,i]))
    np.fill_diagonal(C_clean,1)
    return C_clean    
    
    
def LouvainCorrelationClustering(R):   # R is a matrix of return
    N=R.shape[1]
    T=R.shape[0]


    q=N*1./T
    lambda_plus=(1.+np.sqrt(q))**2

    C=R.corr()
    lambdas, v = LA.eigh(C)
    
    C_s=compute_C_minus_C0(lambdas,v,lambda_plus)
    C_s=np.abs(C_s)
    
    mygraph= nx.from_numpy_matrix(C_s)
    partition = community.community_louvain.best_partition(mygraph)

    DF=pd.DataFrame.from_dict(partition,orient="index")
    return(DF)

def get_states(inputFile = "Data/us-equities_logreturns.feather", \
            start_date = 50, end_date = 100, memory = 50):
    """ GET_STATES Gets the states for the stocks     

    Parameters
    ----------
    inputFile : STRING, optional
        The filename to load data from.
        The default is "Data/us-equities_logreturns.feather".
    start_date : INT, optional
        The date to start getting states. The default is 50.
    end_date : INT, optional
        The date to stop getting states. The default is 100.
    memory : INT, optional
        The number of days to use in memory. The default is 50.

    Returns
    -------
    None.

    """
    
    R = pd.read_feather(inputFile)  
    df_ = pd.DataFrame() # create the empty pandas dataframe
    
    # get the stock states for each day
    for i in range(start_date,end_date):
        df_ = df_.append(LouvainCorrelationClustering(R[i-memory:i]).T)
    
    # add the column headers
    df_.columns = R.columns
    # add the date index column
    df_.insert(0, 'date', range(start_date,end_date))
    
    return df_

if __name__ == "__main__":
    # execute only if run as a script
    data = get_states(start_date = 500, end_date = 600, memory = 50)
    # save to CSV
    data.to_csv('data_500-600_mem50.csv',index=False)