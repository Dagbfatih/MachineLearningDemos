import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df =    pd.DataFrame(data, columns=feature_names)
df["class"] = y

print(df.head())

