import pandas as pd

#Import the Dataset that we want to read (we have to be in the folder where the file is stored)
ap = pd.read_csv("UJIIndoorLoc_B0-ID-01.csv")

# Attribute type
print(ap.dtypes)

# Number of labels
print(ap.groupby('ID').size().sort_values(ascending=False))

# Correlation between the ID and the Class
ap_corr = ap.corr()
ap_corr_ID = abs(ap_corr['ID'][:-1]).sort_values(ascending=False)
print(ap_corr_ID)


# Access Point with no correlation
print(ap['WAP520'].value_counts().sort_index())

print(ap_corr_ID[ap_corr_ID.isnull()])
