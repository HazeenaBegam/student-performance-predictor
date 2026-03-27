import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dataset.csv")

X = data[['Study_Hours','Social_Media_Hours','Sleep_Hours','Attendance','Assignment_Score']]
y = data['Final_Grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = r2_score(y_test, lr_pred)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = r2_score(y_test, dt_pred)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = r2_score(y_test, rf_pred)

print("Linear Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

import matplotlib.pyplot as plt

models = ["Linear Regression","Decision Tree","Random Forest"]
accuracy = [lr_acc, dt_acc, rf_acc]

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")

plt.show()