# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:48:50 2022

@author: Usuario
"""


# In[2]:

#Load data
data = pd.read_csv("creditcard_small.csv")
data['Class'] = data['Class'].astype('category')
data['Class'] = data['Class'].cat.codes

#Data analysis
print(data.info())
print(data.head())

X = data.iloc[:,:-1]
y = data['Class']

# ## Holdout

# In[3]:

# Train/test split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Create classifier
forest = RandomForestClassifier(n_estimators=100)

# Train
forest.fit(X_train, y_train)

#Predict
y_pred = forest.predict(X_test)

# ## Accuracy

# In[4]:
    
print ("Holdout accuracy: {0:.2f}".format(forest.score(X_test, y_test)))

# ## Confusion matrix

# In[5]:
    
print(confusion_matrix(y_test, y_pred))

# ## Classification report

# In[6]:
    
print(classification_report(y_test, y_pred, zero_division = 0))


data = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
data.fit(X_train, y_train)

y_pred = data.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()

y_score = data.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=data.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=data.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()

FP = cm.sum(axis=0) - np.diag(cm)  
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# In[7]:

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print('The TPR is') 
print(TPR)

print('The FPR is') 
print(FPR)

print('The Overall Accuracy is')
print(ACC) 


