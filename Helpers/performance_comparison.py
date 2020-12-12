# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:54:23 2020
Accuracy, F1-score, MAE plot with the three models 

@author: PAHU95377
Modified on : Sat Dec 12 12:44:22 2020

Update : 
    1- delete the plot title since it will be added in latex later on
    2- Adding the dummy classifier
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 

def perf_comp():
    models = {'Model': ['Dummy Classifier','Random Forest', 'LSTM', 'TCN', \
                        'Dummy Classifier','Random Forest', 'LSTM', 'TCN', \
                        'Dummy Classifier','Random Forest', 'LSTM', 'TCN', \
                        'Dummy Classifier','Random Forest', 'LSTM', 'TCN', \
                        'Dummy Classifier','Random Forest', 'LSTM', 'TCN'],
              'Accuracy': [0.12, 0.14, 0.16, 0.17, \
                           0.11, 0.13, 0.16, 0.17, \
                           0.11, 0.14, 0.16, 0.17, \
                           0.11, 0.15, 0.16, 0.17, \
                           0.11, 0.14, 0.16, 0.17],
              'F1-score': [0.08, 0.10, 0.16, 0.17, \
                           0.07, 0.10, 0.16, 0.17, \
                           0.07, 0.10, 0.16, 0.17, \
                           0.07, 0.12, 0.16, 0.17, \
                           0.07, 0.12, 0.16, 0.17],
              'Recall' :  [0.12,0.14,0.12,0.14,\
                           0.11,0.13,0.12,0.14,\
                           0.11,0.14,0.12,0.14,\
                           0.11,0.15,0.12,0.14,\
                           0.11,0.14,0.12,0.14],
              'Days':     [150,150,150,150,\
                           100,100,100,100,\
                            50,50,50,50,\
                            10,10,10,10,\
                             2,2,2,2]
              }
    
    df = pd.DataFrame(models, columns = ['Model', 'Accuracy', 'Recall', 'Days','F1-score'])
    
    print (df)
      
    g = sns.catplot(x = 'Days', y='Accuracy', hue = 'Model',data=df, kind='bar',\
                    palette="ch:s=.25,rot=-.25")
    g.despine(left=True)
    plt.savefig('Figures\overall_accuracy.pdf', dpi=1080)
    
    g = sns.catplot(x = 'Days', y='Recall', hue = 'Model',data=df, kind='bar',\
                    palette="ch:.25") 
    g.despine(left=True)
    plt.savefig('Figures\overall_recall.pdf', dpi =1080)
    
    g = sns.catplot(x = 'Days', y='F1-score', hue = 'Model',data=df, kind='bar',\
                    palette="ch:s=.25,rot=-.25")
    g.despine(left=True)
    plt.savefig('Figures\overall_f1-score.pdf', dpi=1080)
    return