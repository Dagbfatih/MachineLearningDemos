import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cluster 1
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)

# Cluster 2
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)

# Cluster 3
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)

x = np.concatenate((x1, x2, x3), axis=0)

y = np.concatenate((y1, y2, y3), axis=0)

dictionary = {"x": x, "y": y}

data = pd.DataFrame(dictionary)

print(data.head())

# -------------------
# This is the result picture we want to see at the end of our k-mean clustering algorithm run.
# -------------------
# plt.figure()
# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
# plt.scatter(x3,y3)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('the data created for k-mean clustering method example')
# plt.show()

# -------------------
# This is the first data (simulated) without clustering
# -------------------
# plt.scatter(x1,y1, color='black')
# plt.scatter(x2,y2, color='black')
# plt.scatter(x3,y3, color='black')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('the data created for k-mean clustering method example')
# plt.show()

wcss = []

# <<<<<- Elbow method

# To see how many clusters are there in 'data' for k-means
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
# plt.figure()
# plt.plot(range(1, 15), wcss)
# plt.xticks(range(1, 15))
# plt.xlabel("Küme sayısı")
# plt.ylabel("wcss")
# plt.show()

# Elbow method ->>>>>

k_mean = KMeans(n_clusters=3)
clusters = k_mean.fit_predict(data)

data["label"] = clusters

plt.figure()
plt.scatter(
    data.x[data.label == 0], data.y[data.label == 0], color="red", label="Kume 1"
)
plt.scatter(
    data.x[data.label == 1], data.y[data.label == 1], color="green", label="Kume 2"
)
plt.scatter(
    data.x[data.label == 2], data.y[data.label == 2], color="blue", label="Kume 3"
)
plt.scatter(
    k_mean.cluster_centers_[:, 0], k_mean.cluster_centers_[:, 1], color="yellow", label="Centers"
)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("3-mean cluster result")
plt.show()
