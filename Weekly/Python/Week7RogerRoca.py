# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from apyori import apriori



# In[2]:

#Load data
data = pd.read_csv("basket.csv")

data.head()

records = []
for i in range(0, 9834): #Can be done with len too 
    records.append([str(data.values[i,j]) for j in range(0, 31)])
    
association_rules = apriori(records, min_support=0.010, min_confidence=0.2, min_lift=3, min_length=4)


# In[3]:

for item in association_rules:
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    

