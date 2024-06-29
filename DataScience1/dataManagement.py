import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from plotly import py
import warnings
warnings.filterwarnings("ignore")

from models.athlete import Athlete

# Prepare Data
data: DataFrame = pd.read_csv("./database/athletes_clean.csv")

dumpData = data.copy()

dumpData["Team"] = np.where(
    dumpData["Team"].str.contains("Germany"), "Germany", dumpData["Team"]
)

dumpData["Team"] = np.where(
    dumpData["Team"].str.contains("Norway"), "Norway", dumpData["Team"]
)

dumpData["Team"] = np.where(
    dumpData["Team"].str.contains("Britain"), "Britain", dumpData["Team"]
)

newData = dumpData

print(newData)