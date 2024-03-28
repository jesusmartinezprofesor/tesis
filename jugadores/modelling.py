# Carga de librerias
import pandas as pd
from sklearn.cluster import KMeans
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
tabla1_raw = pd.read_excel(xls, "base de datos 1 jugadores")

columnas = [
    "Influenciapaterna",
    "organización",
    "Expectativasallogro",
    "Miedoaloerrores",
    "destre.Aperder",
    "destre.Aganar",
    "dstre.durantejuego",
    "destre.Juegojusto",
    "habilidadessociales",
    "satisvida",
]
tabla1 = tabla1_raw[columnas]

# Estandarizado
st = StandardScaler()
tabla1_st = st.fit_transform(tabla1)
tabla1_st = pd.DataFrame(tabla1_st, columns=columnas)


# BEST MODEL
kmeans_model = KMeans(n_clusters=3)
kmeans_labels = kmeans_model.fit_predict(tabla1_st)


# Definimos X e Y
X = tabla1_st
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
model.save("jugadores_bestmodel.keras")


# Save the scaler to a file
dump(st, "jugadores_scaler.joblib")
