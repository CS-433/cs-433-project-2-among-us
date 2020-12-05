# -*- coding: utf-8 -*-
"""
Performance indicators for a supervised multiclass machine learning

Created by : BenMobility
Created on : Sat Nov 28 11:03:49 2020
Modified on : Sat Dec 05 12:44:20 2020

Info :     
    
UPDATE : 
    1. Create a csv for the performance indicators so we can keep track
you have to add a title        
"""
#%% Librairies
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Confusion matrix
def con_matrix(y_test, y_pred, title):
    """
    CONFUSION MATRIX, Each entry in a confusion matrix denotes the number of 
    predictions made by the model where it classified the classes correctly or incorrectly
    
    Parameters
    ----------
    y_test : np.array n x 1. integer Labels of each day.
    y_pred : np.array n x 1. integer predicted labels of each day.
    title : str title figures

    Returns
    -------
    confusion : pandas dataframe confusion matrix m x m where m is the number of unique labels.
    prints the confusion matrix 
    """
    #if y_test and y_pred are np.array
    if type(y_test) == np.ndarray:
        y_test = y_test.tolist()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.tolist()
    #for labels graph
    labels = y_pred.copy()
    for i in range(len(y_pred)):
        if labels[i] != y_test[i]:
            labels.append(y_test[i])
    labels = sorted(list(dict.fromkeys(labels)))
    
    #confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix\n')
    print(confusion)
    
    #plotting the confusion matrix
    df_cm = pd.DataFrame(confusion,labels,labels)
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, cmap='Blues') # font size
    plt.yticks(rotation=0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    filename = 'Figures/{}.pdf'.format(title)
    plt.savefig(filename, dpi = 1080)
    return df_cm

#%%Performance indicators

def p_inds(y_test, y_pred, title):
    """
    Compute the performance indicators F1-score, accuracy, recall, precision
    Prints the macro, micro and weighted performance indicators
    Parameters
    ----------
    y_test : np.array n x 1. integer Labels of each day.
    y_pred : np.array n x 1. integer predicted labels of each day.

    Returns
    -------
    None.
    csv file with prediction, recall, f1-score for each states and overall 
    that includes also accuracy

    """
    #if y_test and y_pred are np.array
    if type(y_test) == np.ndarray:
        y_test = y_test.tolist()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.tolist()
    #for labels graph
    labels = y_pred.copy()
    for i in range(len(y_pred)):
        if labels[i] != y_test[i]:
            labels.append(y_test[i])
    labels = sorted(list(dict.fromkeys(labels)))
    names = []
    for i in range(len(labels)):
        names.append('{}'.format(labels[i]))
    print('Model Performance\n')
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=names))
    class_dict = classification_report(y_test, y_pred, target_names=names, \
                                       output_dict=True)
    class_d = pd.DataFrame.from_dict(class_dict)
    filename = ('Data/{}.csv'.format(title))
    class_d.to_csv(filename)
    return
