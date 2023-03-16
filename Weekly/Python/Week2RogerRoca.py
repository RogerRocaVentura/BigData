# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:38:40 2022

@author: Usuario
    """

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

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



