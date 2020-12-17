# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:54:23 2020
Accuracy, F1-score, MAE plot with the three models 

@author: PAHU95377
Modified on : Sat Dec 12 12:44:22 2020

Update : 
    1- delete the plot title since it will be added in latex later on
    2- Adding the MK classifier
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def perf_comp(df):
    # load data manually, normally commented
    #in_file_path = './Data/Results.csv'
    #df = pd.read_csv(in_file_path)
    
    plot_data = df.loc[df['Model'] != 'MK'] # remove Markov from bars
    plt.close('all') # close open graphs
    
    # ACCURACY PLOT
    g = sns.catplot(x = 'Memory', y='accuracy', hue = 'Model',data=plot_data, \
                    kind='bar', palette="ch:s=.25,rot=-.25")
    g.despine(left=True)
    
    plt.axhline(df.loc[(df['Model'] == 'MK'),'accuracy'].values)
    plt.text(5, df.loc[(df['Model'] == 'MK'),'accuracy'], 'MK', \
             fontsize=9, va='center', ha='center')
    
    plt.savefig('Figures\overall_accuracy.pdf', dpi=1080)
    
    
    # RECALL PLOT
    g = sns.catplot(x = 'Memory', y='recall', hue = 'Model',data=plot_data, \
                    kind='bar', palette="ch:.25") 
    g.despine(left=True)
    
    plt.axhline(df.loc[(df['Model'] == 'MK'),'recall'].values)
    plt.text(5, df.loc[(df['Model'] == 'MK'),'recall'], 'MK', \
             fontsize=9, va='center', ha='center')
        
    plt.savefig('Figures\overall_recall.pdf', dpi =1080)
    
    
    # F1 PLOT
    g = sns.catplot(x = 'Memory', y='f1', hue = 'Model',data=plot_data, \
                    kind='bar', palette="ch:s=.25,rot=-.25")
    g.despine(left=True)
    
    plt.axhline(df.loc[(df['Model'] == 'MK'),'f1'].values)
    plt.text(5, df.loc[(df['Model'] == 'MK'),'f1'], 'MK', \
             fontsize=9, va='center', ha='center')
        
    plt.savefig('Figures\overall_f1-score.pdf', dpi=1080)
    
    
    # PRECISION PLOT
    g = sns.catplot(x = 'Memory', y='precision', hue = 'Model',data=plot_data,\
                    kind='bar', palette="ch:s=.25,rot=-.25")
    g.despine(left=True)
    
    plt.axhline(df.loc[(df['Model'] == 'MK'),'precision'].values)
    plt.text(4, df.loc[(df['Model'] == 'MK'),'precision'], 'MK', \
             fontsize=9, va='center', ha='center')
        
    plt.savefig('Figures\overall_precision.pdf', dpi=1080)
    
if __name__ == "__main__":
    perf_comp(1)