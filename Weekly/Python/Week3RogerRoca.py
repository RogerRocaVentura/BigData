# In[1]:
    
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

# In[2]:

data = pd.read_csv("Telecom_Churn.csv") 
data['Churn'] = data['Churn'].astype('category')
data['Churn'] = data['Churn'].cat.codes

data.drop('State', inplace=True, axis=1)
data.drop('International plan', inplace=True, axis=1)
data.drop('Voice mail plan', inplace=True, axis=1)


X = data.iloc[:,:-1]
y = data['Churn']

# ## Holdout

# In[3]:

# Train/test split    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Create classifier
forest = RandomForestClassifier(n_estimators=10)

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

# In[7]:
    
report = classification_report(y_test, y_pred, zero_division = 0, output_dict = True)
print(report['accuracy'])
print(report['weighted avg']['precision'])
print(report['weighted avg']['recall'])
print(report['weighted avg']['f1-score'])

# In[8]:
    
precision_recall_fscore_support(y_test, y_pred, average='weighted')

# ## Cross-validation

# In[9]:
    
#Create classifier
cv_forest = RandomForestClassifier(n_estimators=10)

#ID-fold cross-validation
cv_scores = cross_val_score(cv_forest, X, y, cv=10)
print("Cross-validation: ", cv_scores)

#Print the mean
print("accuracy: {0:0.2f} (+/- {1:0.2f})".format(cv_scores.mean(), cv_scores.std() * 2))

# In[10]:
    
names = ["kNN", "DT", "NB", "SVC", "MLP", "RF"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(),
    MLPClassifier(max_iter=1000),
    RandomForestClassifier(n_estimators=10)]

metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'Train (s)', 'Pred (s)'], index=names)

for name, clf in zip(names,classifiers):
    
    #Train
    t0=time.time()
    clf.fit(X_train, y_train)
    t1=time.time()
    
    
   
        
    #Test
    y_pred = clf.predict(X_test)
    t2=time.time()
 
    #Metrics
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    acc = round(report['accuracy'], 2)
    pro = round(report['weighted avg']['precision'],2)
    rec = round(report['weighted avg']['recall'], 2)
    f1s = round(report['weighted avg']['f1-score'], 2)
    t01 = round(t1-t0, 2)
    t12 = round(t2-t1, 2)

    
    metrics.at[name] = [acc, pro, rec, f1s, t01, t12]

metrics


# In[11]:
    
print(f"Training time: {t1 - t0}s")
print(f"Test time: {t2 - t0}s")
    
#Load dataset
data = pd.read_csv("Telecom_Churn.csv") 
data['Churn'] = data['Churn'].astype('category')
data['Churn'] = data['Churn'].cat.codes


data.drop('State', inplace=True, axis=1)
data.drop('International plan', inplace=True, axis=1)
data.drop('Voice mail plan', inplace=True, axis=1)


X = data.iloc[:,:-1]
y = data['Churn']


# Train/test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

#Evaluate classifiers

names = ["kNN", "DT", "NB", "SVC", "MLP", "RF"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(),
    MLPClassifier(max_iter=1000),
    RandomForestClassifier(n_estimators=10)]

metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'Train (s)', 'Pred (s)'], index=names)

for name, clf in zip(names,classifiers):
    
    #Train
    t0=time.time()
    clf.fit(X_train, y_train)
    t1=time.time()
    
    
    y_pred = clf.predict(X_test)
    t2=time.time()
   
    #Metrics
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    acc = round(report['accuracy'], 2)
    pro = round(report['weighted avg']['precision'],2)
    rec = round(report['weighted avg']['recall'], 2)
    f1s = round(report['weighted avg']['f1-score'], 2)
    t01 = round(t1-t0, 2)
    t12 = round(t2-t1, 2)
    
    metrics.at[name] = [acc, pro, rec, f1s, t01, t12]

metrics


# In[12]:
print(f"Training time: {t1 - t0}s")
print(f"Test time: {t2 - t0}s")
#Load dataset
data = pd.read_csv("Telecom_Churn.csv") 
data['Churn'] = data['Churn'].astype('category')
data['Churn'] = data['Churn'].cat.codes

data.drop('State', inplace=True, axis=1)
data.drop('International plan', inplace=True, axis=1)
data.drop('Voice mail plan', inplace=True, axis=1)

X = data.iloc[:,:-1]
y = data['Churn']

# Train/test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Evaluate classifiers

names = ["kNN", "DT", "NB", "SVC", "MLP", "RF"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(),
    MLPClassifier(max_iter=1000),
    RandomForestClassifier(n_estimators=10)]

metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-score', 'Train (s)', 'Pred (s)'], index=names)

for name, clf in zip(names,classifiers):
    try:
        t0=time.time()
        clf.fit(X_train, y_train)
        t1=time.time()
       
        #Test
        y_pred = clf.predict(X_test)
        t2=time.time()
 
        #Metrics
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        acc = round(report['accuracy'], 2)
        pro = round(report['weighted avg']['precision'],2)
        rec = round(report['weighted avg']['recall'], 2)
        f1s = round(report['weighted avg']['f1-score'], 2)
        t01 = round(t1-t0, 2) 
        t12 = round(t2-t1, 2)
      
        metrics.at[name] = [acc, pro, rec, f1s, t01, t12]
        
    except ValueError:
        pass
   
metrics
# In 13 []:
print(f"Training time: {t1 - t0}s")
print(f"Test time: {t2 - t0}s")
