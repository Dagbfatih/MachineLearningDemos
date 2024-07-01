import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/coklu_dogrusal_regresyon_veriseti.csv", sep=";")
print(df.head())
print("\n----------\n")

x = df.iloc[:, [0, 2]].values
print("x:\n{}".format(x) + "\n\n----------\n")

y = df.maas.values.reshape(-1, 1)
print("y:\n{}".format(y) + "\n\n----------\n")

multiLinearRegression = LinearRegression()

multiLinearRegression.fit(x, y)

test1 = multiLinearRegression.predict([[10, 35]])  # deneyim, yas
testResult1 = np.int64(test1[0, 0])  # Remove decimals
print("Test result 1: {}".format(testResult1))
