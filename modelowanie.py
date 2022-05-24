#!/usr/bin/env python
# coding: utf-8

# In[21]:


from funkcja_perceptron import Perceptron 
import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris
import pickle

def modelowanie():

    baza = load_iris()
    baza1 = pd.DataFrame(data = baza['data'], columns = baza['feature_names'])
    baza1['Target'] = baza['target']
    baza1 = baza1.iloc[:,[0,2,4]]
    #print(baza1)

    baza2 = Perceptron()
    baza2.fit(baza1.iloc[:,[0,1]].values,baza1.iloc[:,2].values)

    with open('model.pkl', 'wb') as zapis:
        pickle.dump(baza2,zapis)

