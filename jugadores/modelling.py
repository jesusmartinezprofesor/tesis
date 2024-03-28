# Carga de librerias
import pandas as pd
from sklearn.cluster import KMeans
import sys
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report


#pyhton version
print("Version de Python: ", sys.version)

xls = pd.ExcelFile("data.xlsx")
tabla1_raw = pd.read_excel(xls, 'base de datos 1 jugadores')
print(tabla1_raw.shape)
tabla1_raw.head()

columnas = ["Influenciapaterna", 	"organización", 	"Expectativasallogro",	"Miedoaloerrores",	"destre.Aperder",	"destre.Aganar", 	"dstre.durantejuego",	"destre.Juegojusto", 	"habilidadessociales",	"satisvida"]
tabla1 = tabla1_raw[columnas]

#Estandarizado 
st = StandardScaler()
tabla1_st = st.fit_transform(tabla1)
tabla1_st = pd.DataFrame(tabla1_st, columns=columnas)
tabla1_st


# BEST MODEL
kmeans_model = KMeans(n_clusters=3)
kmeans_labels = kmeans_model.fit_predict(tabla1_st)


# Assuming you have your dataset with features and labels
# X contains the features and y contains the labels
# Replace X and y with your actual dataset
X = tabla1_st  # Example data with 100 samples and 10 features
y = kmeans_labels # Example labels (0, 1, or 2) for 100 samples

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))  # Input layer with 10 features
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 units
model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for classification

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)



loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Generate predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# model
model.save("jugadores_bestmodel.keras")

# StandardScaler

# Save the scaler to a file
dump(st, 'jugadores_scaler.joblib')
