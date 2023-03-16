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
data = pd.read_csv("customers_clustering.csv")
data.head()

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
# Visualize data point
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c="blue")
plt.xlabel("Income")
plt.ylabel("Spending score")
plt.show()

# In[3]:
K = 4;

Centroids = (X.sample(n=K))
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c="blue")
plt.scatter(Centroids["Annual Income (k$)"], Centroids["Spending Score (1-100)"], c="red")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# Step 3 - Assign all the points to the closest cluster centroid
# Step 4 - Recompute centroids of newly formed clusters
# Step 5 - Repeat step 3 and 4

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1, row_c in Centroids.iterrows():
        ED=[]
        for index2, row_d in XD.iterrows():
            d1 = (row_c["Annual Income (k$)"]-row_d["Annual Income (k$)"])**2
            d2 = (row_c["Spending Score (1-100)"]-row_d["Spending Score (1-100)"])**2
            d = sqrt(d1+d2)
            ED.append(d)
        X[i] = ED
        i = i+1
    
    C = []
    for index, row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Spending Score (1-100)", "Annual Income (k$)"]]
    if j == 0:
        diff = 1
        j = j+1
    else:
        diff = (Centroids_new['Spending Score (1-100)'] - Centroids['Spending Score (1-100)']).sum() + (Centroids_new['Annual Income (k$)'] - Centroids['Annual Income (k$)']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Spending Score (1-100)","Annual Income (k$)"]]
    
    
color=['blue','green','cyan','yellow']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["Annual Income (k$)"],data["Spending Score (1-100)"],c=color[k])
plt.scatter(Centroids["Annual Income (k$)"],Centroids["Spending Score (1-100)"],c='red')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

