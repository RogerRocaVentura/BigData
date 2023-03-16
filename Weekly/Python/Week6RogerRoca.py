# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:33:02 2022

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:48:50 2022

@author: Usuario


"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




# In[2]:

#Load data
data = pd.read_csv("games.csv")
data.head()




