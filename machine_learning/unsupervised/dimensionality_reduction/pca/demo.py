import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.decomposition import PCA

# Principal Component Analysis

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns=feature_names)
df["class_"] = y

# print(df.head())

pca = PCA(n_components=2, whiten=True)
pca.fit(data)

x_pca = pca.transform(data)

print("Variance ratio: ", pca.explained_variance_ratio_)
print("Sum: ", sum(pca.explained_variance_ratio_))

# Data visualization

df["p1"] = x_pca[:, 0]
df["p2"] = x_pca[:, 1]

color = ["red", "green", "blue"]

for each in range(3):
    plt.scatter(
        df.p1[df.class_ == each],
        df.p2[df.class_ == each],
        color=color[each],
        label=iris.target_names[each],
    )

plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
