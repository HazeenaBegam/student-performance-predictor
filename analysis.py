import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

corr = data.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()

plt.scatter(data["Social_Media_Hours"], data["Final_Grade"])

plt.title("Social Media Hours vs Final Grade")

plt.xlabel("Social Media Hours")

plt.ylabel("Final Grade")

plt.show()