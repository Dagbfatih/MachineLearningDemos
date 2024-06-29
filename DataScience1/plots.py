import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import warnings

from models.athlete import Athlete

warnings.filterwarnings("ignore")

data: DataFrame = pd.read_csv("./database/athletes_clean.csv")

print(data.info())


def plotHistogram(columnName: str):
    plt.figure()
    plt.hist(data[columnName], bins=50, color="orange")
    plt.xlabel(columnName)
    plt.ylabel("Frequence")
    plt.title("Data Frequency - {}".format(columnName))
    plt.show()


def plotBox(columnName: str):
    plt.boxplot(data[columnName])
    plt.xlabel("Value")
    plt.ylabel(columnName)
    plt.title("Box Plot For - {}".format(columnName))
    plt.show()

# --- Tabloları yazdır ---
# for i in ["Age", "Weight", "Height", "Year"]:
#     plotHistogram(i)
# plotBox("Age")

# --- corr(): bu 3 sütunun birbiriyle ne kadar uyumlu artıp azaldığını gösterir ---
# print(data.loc[:, ["Age", "Height", "Weight"]].corr())


# --- Medal sütunlarını üç ana sütuna ayır ---
# dumpData = data.copy()

# dumpData = pd.get_dummies(dumpData, columns=["Medal"])
# print(dumpData.head(4))