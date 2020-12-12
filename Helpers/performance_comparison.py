# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:54:23 2020
Accuracy, F1-score, MAE plot with the three models 

@author: PAHU95377
Modified on : Sat Dec 5 12:44:22 2020

Update : 
    1- delete the plot title since it will be added in latex later on
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 


models = {'Model': ['Random Forest', 'TCN', 'NN','Random Forest', 'TCN', 'NN', 'Random Forest', 'TCN', 'NN','Random Forest', 'TCN', 'NN','Random Forest', 'TCN', 'NN'],
          'Accuracy': [0.33, 0.44, 0.55,  0.33, 0.44, 0.55,  0.6,0.7,0.8,  0.6,0.7,0.8,  0.8,0.9,0.95],
          'F1-score': [0.33, 0.44, 0.55,  0.33, 0.44, 0.55,  0.6,0.7,0.8,  0.6,0.7,0.8,  0.8,0.9,0.95],
          'MAE' : [3,4,5, 3,4,5, 3,4,5, 2,3,4 ,1,2,3],
          'Days':[150,150,150, 100,100,100, 50,50,50, 20,20,20, 10,10,10]
          }

df = pd.DataFrame(models, columns = ['Model', 'Accuracy', 'MAE', 'Days','F1-score'])

ratio1 = 'acc_ra_80'
ratio2 = 'mae_ra_80'
ratio3 = 'f1s_ra_80'

print (df)


fig = plt.figure()
g = sns.catplot(x = 'Days', y='Accuracy', hue = 'Model',data=df, kind='bar',\
                palette="ch:s=.25,rot=-.25")
g.despine(left=True)
plt.savefig('Figures\{}.pdf'.format(ratio1), dpi=1080)


fig = plt.figure()
g = sns.catplot(x = 'Days', y='MAE', hue = 'Model',data=df, kind='bar',\
                palette="ch:.25") 
g.despine(left=True)
plt.savefig('Figures\{}.pdf'.format(ratio2), dpi =1080)

fig = plt.figure()

g = sns.catplot(x = 'Days', y='F1-score', hue = 'Model',data=df, kind='bar',\
                palette="ch:s=.25,rot=-.25")
g.despine(left=True)
plt.savefig('Figures\{}.pdf'.format(ratio3), dpi=1080)