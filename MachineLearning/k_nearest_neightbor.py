import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("data/ortopedik_hastalarìn_biyomekanik_özellikleri_20220209.csv")

# print(data.head())

# sns.scatterplot(
#     data=data, x="lumbar_lordosis_angle", y="pelvic_tilt numeric", hue="class"
# )
# plt.xlabel("lumbar lordosis angle")
# plt.ylabel("pelvic tilt")
# plt.legend()
# plt.show()

data["class"] = data["class"].apply(lambda x: 1 if x == "Abnormal" else 0)

y = data["class"].values

x_data = data.drop(["class"], axis=1)

# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=1
)

neighbor_count = 29
knn = KNeighborsClassifier(n_neighbors = neighbor_count)
knn.fit(x_train, y_train)

prediction = knn.predict(x_test)
print("Prediction accuracy: {0}".format(knn.score(x_test, y_test)))
