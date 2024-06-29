import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Verileri yükle
dataframe = pd.read_csv("data/data_yaren2.csv", delimiter=";", encoding="utf-8")

print(dataframe.head())
print("---------------------------------\n---------------------------------")

# Deneme ismi sütununu sil
dataframe.drop(["DENEME"], axis=1, inplace=True)

# Sıra ve puan sütunlarını sil (İstersen sütun ismiyle, istersen index ile)
dataframe.drop(["PUAN"], axis=1, inplace=True)
dataframe.drop(dataframe.columns[[-1, -3, -4, -5]], axis=1, inplace=True)

# print(dataframe.head())

# Netler ve yüzdeliği virgül değil nokta ile ayır
turkce_column_name = "TURKCE_N"
ink_column_name = "INK_N"
din_column_name = "DIN_N"
ing_column_name = "ING_N"
mat_column_name = "MAT_N"
fen_column_name = "FEN_N"
toplam_column_name = "TOPLAM_N"
yuzdelik_column_name = "YUZDELIK"

dataframe[turkce_column_name] = dataframe[turkce_column_name].str.replace(",", ".")
dataframe[turkce_column_name] = pd.to_numeric(
    dataframe[turkce_column_name], errors="raise"
)

dataframe[ink_column_name] = dataframe[ink_column_name].str.replace(",", ".")
dataframe[ink_column_name] = pd.to_numeric(dataframe[ink_column_name], errors="raise")

dataframe[din_column_name] = dataframe[din_column_name].str.replace(",", ".")
dataframe[din_column_name] = pd.to_numeric(dataframe[din_column_name], errors="raise")

dataframe[ing_column_name] = dataframe[ing_column_name].str.replace(",", ".")
dataframe[ing_column_name] = pd.to_numeric(dataframe[ing_column_name], errors="raise")

dataframe[mat_column_name] = dataframe[mat_column_name].str.replace(",", ".")
dataframe[mat_column_name] = pd.to_numeric(dataframe[mat_column_name], errors="raise")

dataframe[fen_column_name] = dataframe[fen_column_name].str.replace(",", ".")
dataframe[fen_column_name] = pd.to_numeric(dataframe[fen_column_name], errors="raise")

dataframe[toplam_column_name] = dataframe[toplam_column_name].str.replace(",", ".")
dataframe[toplam_column_name] = pd.to_numeric(
    dataframe[toplam_column_name], errors="raise"
)

dataframe[yuzdelik_column_name] = dataframe[yuzdelik_column_name].str.replace(",", ".")
dataframe[yuzdelik_column_name] = pd.to_numeric(
    dataframe[yuzdelik_column_name], errors="raise"
)

# print(dataframe.head())

# Scaler oluştur ve veriyi dönüştür (MinMax scaler.fit-transform)

scaler = MinMaxScaler()

dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns, index=dataframe.index)

# dataframe.to_csv("data/data_yaren_ready.csv")

print(dataframe.head())

# Dataframeden son sütunu al ve array olarak ata
y = dataframe[dataframe.columns[-1]].values

# Dataframeden son sütunu çıkar
x = dataframe.drop(dataframe.columns[[-1]], axis=1)

print(x.head())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=None, shuffle=False
)

model = Sequential([
    Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # No activation function for output layer (assuming it's regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print("Test Loss:", loss)

predictions = model.predict(x_test)

predarray = []
for pred in predictions:
    predarray.append(pred[0])

x_test["YUZDELIK"] = predarray
predictions_original_scale = scaler.inverse_transform(x_test)


# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()