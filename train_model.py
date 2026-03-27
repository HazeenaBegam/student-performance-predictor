import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("dataset.csv")

X = data[['Study_Hours','Social_Media_Hours','Sleep_Hours','Attendance','Assignment_Score']]
y = data['Final_Grade']

model = RandomForestRegressor()

model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")