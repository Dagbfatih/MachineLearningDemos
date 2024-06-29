import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("athletes.csv")

data.info()

print(data.columns)

data.rename(columns={
    'ID':'Id',
    'Name':'Name',
    'Sex':'Gender',
    'Age':'Age',
    'Height':'Height',
    'Weight':'Weight',
    'Team':'Team',
    'NOC':'Noc',
    'Games':'Games',
    'Year':'Year',
    'Season':'Season',
    'City':'City',
    'Sport':'Sport',
    'Event':'Event',
    'Medal':'Medal',
    }, inplace=True)

print();print()
print(data.head(2))

data = data.drop(["Id", "Games"], axis=1)

uniqueEvents = pd.unique(data.Event)

print("Unique values: {}".format(len(uniqueEvents)))

# her bir eventi iteratif olarak dolaş
# event özelinde height ye weight ortalamalarını hesapla
# event özelinde kayıp height ye weight değerlerini 
# etkinlik ortalamalarına eşitle 

tempData=data.copy()

for event in uniqueEvents:
    
    filteredEvents = tempData.Event == event
    
    filteredData = tempData[filteredEvents]
    
    for s in ["Height", "Weight"]:
        average = np.round(
            np.mean(filteredData[s]),
            2)
        
        if ~np.isnan(average):
            filteredData[s] = filteredData[s].fillna(average)
        else:
            averageOfAllData = np.round(np.mean(data[s]), 2)
            filteredData[s] = filteredData[s].fillna(averageOfAllData)
    
    tempData[filteredEvents] = filteredData

data = tempData.copy()
print(data.info())    
        
# fill the Age column nan values

averageOfAges = np.round(np.mean(data.Age), 2)

print("Average of age {}".format(averageOfAges))

data["Age"] = data["Age"].fillna(averageOfAges)

print(data.info())

medal_column = data["Medal"]
pd.isnull(medal_column)

filterOfAthletesHaveMedal = ~pd.isnull(medal_column)
data = data[filterOfAthletesHaveMedal]

print(len(data))

data.to_csv("athletes_clean.csv", index = False)
