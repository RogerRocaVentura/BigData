# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:38:40 2022

@author: Usuario
    """

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# Load iris dataset
data = pd.read_csv("Telecom_Churn.csv") 
data['Churn'] = data['Churn'].astype('category')
data['Churn'] = data['Churn'].cat.codes
print(data.info)
print(data.info())


# Attribute type

print(data.dtypes)

# Number of labels
print(data.groupby('Churn').size().sort_values(ascending=False))

# Correlation
data_corr = data.corr()
data_corr_Churn = abs(data_corr['Churn'][:-1]).sort_values(ascending=False)
print(data_corr_Churn)


data.head(2)
plt.figure(figsize=(12,8))
sns.heatmap(data_corr, cmap="rainbow_r",annot=True)


# Split in train and test datasets
# 2D Attributes
X = data.drop(['Churn'], axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
labels =  np.unique(y)
labels_count = np.bincount(y)
labels_train_count = np.bincount(y_train)
labels_test_count = np.bincount(y_test)

names = ["Total day charge", " Customer Service Calls", "Total day minutes", "Number vmail messages", "Total eve charge", "Total intl calls", "Total intl minutes", "Total eve minutes", "Total intl charge", "Total night minutes", "Total night charge", "Total eve calls", "Area code", "Total night calls", "Account length", "Total day calls"]

X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)


# ## Classifiers


### KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy (KNeighborsClassifier): {0:.2f}".format(acc))

###  DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy (DecisionTreeClassifier): {0:.2f}".format(acc))

### GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy (GaussianNB): {0:.2f}".format(acc))

### SVC
clf = SVC()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy (SVC): {0:.2f}".format(acc))

### MLPClassifier
clf = MLPClassifier()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy (MLPClassifier): {0:.2f}".format(acc))

### MLPClassifier
clf = MLPClassifier(max_iter=175)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy (MLPClassifier): {0:.2f}".format(acc))


# ### Using a loop

names = ["kNN", "DT", "NB", "SVC", "MLP"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(),
    MLPClassifier(max_iter=175)]

metrics = pd.DataFrame(columns=['Accuracy', 'Train cost (s)', 'Pred cost (s)'], index=names)

for name, clf in zip(names, classifiers):
    t0=time.time()
    clf.fit(X_train, y_train)
    t1=time.time()
    metrics.at['Accuracy', name] = clf.score(X_test, y_test)
    #acc = round(clf.score(X_test, y_test), 2)
    t2=time.time()
    metrics.at['Train cost', name] = round(t1-t0, 3)
    metrics.at['Prediction cost', name] = round(t2-t1, 3)
    #t01 = round(t1-t0, 3)
    #t12 = round(t2-t1, 3)
    #metrics.at[name] = [acc, t01, t12]
    
metrics



