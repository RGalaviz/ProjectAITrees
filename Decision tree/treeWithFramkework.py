import pandas as pd
import numpy as np

from sklearn.utils import shuffle


columns = ["Index","Refractive Index","Sodium","Magnesium","Aluminum","Silicon","Potassium","Calcium","Barium","Iron","Type of glass"]

data = pd.read_csv('./glass-1.csv',names=columns)

print(data.head())

#take only the not index columns

data = data[["Refractive Index","Sodium","Magnesium","Aluminum","Silicon","Potassium","Calcium","Barium","Iron","Type of glass"]]

print(data.head())

#shuffle our dataset in order to have a better way of splitting it in test set and training set

my_df = shuffle(data)

my_df.reset_index(inplace=True, drop=True)

print(my_df.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=50)

from sklearn import tree
model = tree.DecisionTreeClassifier()

fitness = model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))
# [1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00,]
print("Predict for: [1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00,]")
print("Class predited:",model.predict([[1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00]]))

print("Predict for: [1.52320,12.78,0.00,1.56,73.36,0.00,8.62,1.67,0.00]")
print("Class predited:", model.predict([[1.51711,12.78,0.00,1.56,73.36,0.00,10.06,1.67,0.00]]))
