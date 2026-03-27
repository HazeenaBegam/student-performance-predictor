import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("dataset.csv")

X = data[['Study_Hours','Social_Media_Hours','Sleep_Hours','Attendance','Assignment_Score']]
y = data['Final_Grade']

model = RandomForestRegressor()
model.fit(X,y)

importance = model.feature_importances_

plt.bar(X.columns, importance)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.show()