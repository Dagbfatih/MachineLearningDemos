import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("data/universite_siralamasi_20220204.csv")
# print(df.head())
df["female_male_ratio"] = df["female_male_ratio"].str.split(":")
df["female_male_ratio"] = df["female_male_ratio"].apply(
    lambda x: x[0] if isinstance(x, list) else None
)

# Convert the column to integers
df["female_male_ratio"] = pd.to_numeric(df["female_male_ratio"], errors="coerce")
df["female_male_ratio"].fillna(0, inplace=True)
df["female_male_ratio"] = df["female_male_ratio"].astype(int)

av = np.round(np.mean(df.female_male_ratio), 1)
print("av: ", av)
df["female_male_ratio"] = df["female_male_ratio"].fillna(av)
print(df["female_male_ratio"])


plt.scatter(df.female_male_ratio, df.year)
plt.xlabel("Kadın oranı")
plt.ylabel("Yıl")
plt.title("Kadın Oranı ve Yıl ilişkisi")
plt.grid(True)
plt.show()
