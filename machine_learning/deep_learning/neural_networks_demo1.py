from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Derin öğrenmede keras kullanacağız ama sklearn bize test eğitim bölünmesini sağlayacak

# Veriyi içe aktaralım
veri = pd.read_csv("machine_learning/data/egitim.csv")
# 28*28 pixel düzleştirilmiş ve 784 sütun
print("Verinin Boyutu: ", veri.shape)
veri.head()

# Sadece etiketi bir ve sıfır olanları alalım
label_filtre0 = 0  # etiketi sıfır
label_filtre1 = 1  # etiketi bir
 
# Etiketi sıfır ve bir olanları filtreleyip birleştirelim
veri = pd.concat([veri[veri["label"] == label_filtre0],
                  veri[veri["label"] == label_filtre1]], axis=0)

veri.head()
