import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble


data = pd.read_csv("kc_house_data.csv")
data.head()

data.describe()

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of bedrooms in decreasing order ')
plt.xlabel('Bedrooms')
plt.ylabel('Houses')
sns.despine

plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, height=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine

plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")
plt.show()

plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")
plt.show()

plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")
plt.show()

plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])

plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")

train1 = data.drop(['id', 'price'],axis=1)

train1.head()

data.floors.value_counts().plot(kind='bar')

plt.scatter(data.floors,data.price)

plt.scatter(data.condition,data.price)

plt.scatter(data.zipcode,data.price)
plt.title("Pricey location by Zipcode")


reg = LinearRegression()
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)




x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)


clf = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'squared_error')
clf.fit(x_train, y_train)

print('The Accuracy of our model is:', clf.score(x_test,y_test))



