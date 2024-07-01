import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

dataFrame = pd.read_csv("data/dogrusal_regresyon_veriseti.csv", sep=";")
print(dataFrame.head())

# plt.scatter(dataFrame.deneyim, dataFrame.maas)
# plt.xlabel("Deneyim (Yıl)")
# plt.ylabel("Maaş (TL)")
# plt.title("Deneyim Maaş İlişkisi")
# plt.grid(True)
# plt.show()

linearReg = LinearRegression()

x = dataFrame.deneyim.values.reshape(-1, 1)
y = dataFrame.maas.values.reshape(-1, 1)

linearReg.fit(x, y)

b0 = linearReg.intercept_ # 0 yıl deneyimi olanbir kişi b0 kadar maaş alır demek. Y eksenini kestiği nokta (yani x = 0)

print("b0: {}".format(b0))

b1 = linearReg.coef_

print("b1: {}".format(b1)) # slope (eğim) - her 1 yıllık deneyim artışı b1 kadar maaş artışına sebep olur.

prediction = linearReg.predict([[20]])
print("Prediction: {}".format(prediction))