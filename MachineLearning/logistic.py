import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("data/ortopedik_hastalarìn_biyomekanik_özellikleri_20220209.csv")

# print(data.head())


data["class"] = data["class"].apply(lambda x: 1 if x == "Abnormal" else 0)

print(data.info())

y = data["class"].values

x_data = data.drop(["class"], axis=1)

sns.pairplot(x_data)
# plt.show()

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42
)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

# print("""
# x train: {0},
# x test: {1},
# y train: {2},
# y test: {3},
# """.format(
#         x_train.shape,
#         x_test.shape,
#         y_train.shape,
#         y_test.shape))

lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)

test_accuracy = lr.score(x_test.T, y_test.T)

print("Test doğruluğu: {}".format(test_accuracy))
