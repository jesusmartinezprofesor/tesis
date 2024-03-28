# Carga de librerias
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import sys
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# StandardScaler
from joblib import dump

# pyhton version
print("Version de Python: ", sys.version)

xls = pd.ExcelFile("../data.xlsx")
tabla2_raw = pd.read_excel(xls, "base de datos 2 niños")

columnas = [
    "Autovaloración",
    "Autoexperiencia",
    "PrexiónExterna",
    "ExperienciaPersonal",
    "Compañeros",
    "Profesor",
    "Positivos",
    "Negativos",
]
tabla2 = tabla2_raw[columnas]

# Estandarizado
st = StandardScaler()
tabla2_st = st.fit_transform(tabla2)
tabla2_st = pd.DataFrame(tabla2_st, columns=columnas)


# BEST MODEL
model = AgglomerativeClustering(n_clusters=3)
kmeans_labels = model.fit_predict(tabla2_st)


# Definimos X e Y
X = tabla2_st
y = kmeans_labels

# Se hace la partición de los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Definimos el modelo neuronal
model = Sequential()
model.add(Dense(64, input_dim=10, activation="relu"))  # Input layer with 10 features
model.add(Dense(32, activation="relu"))  # Hidden layer with 32 units
model.add(
    Dense(3, activation="softmax")
)  # Output layer with 3 units for classification


# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


# model
model.save("ninos_bestmodel.keras")

# Save the scaler to a file
dump(st, "ninos_scaler.joblib")
