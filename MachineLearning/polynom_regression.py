import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("data/polinomsal_regresyon_veriseti_20220209.csv", sep=";")

print(df.head())

x = df.araba_fiyat.values.reshape(-1, 1)
y = df.araba_max_hiz.values.reshape(-1, 1)

plt.scatter(x, y)
plt.xlabel("Araba max hızı")
plt.ylabel("Fiyatı")
plt.title("Araba hız ve fiyat ilişkisi")
plt.grid(True)
# plt.show()

# Polynomical Regression
degree = int(input("Write degree: "))  # ideal 4
polynomial_regression = PolynomialFeatures(degree=degree)

x_polynom = polynomial_regression.fit_transform(x)
print("-------------")
print(x_polynom)

# linear regression fitting
lr = LinearRegression()
lr.fit(x_polynom, y)
y_prediction = lr.predict(x_polynom)

lr.fit(x, y)
y_prediction_linear = lr.predict(x)

plt.scatter(x, y)
plt.plot(x, y_prediction, color="red", label="Polynomical")
plt.plot(x, y_prediction_linear, color="green", label="Reals")
plt.legend()
plt.ylabel("Araba max hızı")
plt.xlabel("Fiyatı")
plt.title("Araba hız ve fiyat ilişkisi")
plt.grid(True)
plt.show()
