# Carga de librerias
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
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
tabla2_raw = pd.read_excel(xls, 'base de datos 2 niños')
tabla2_raw.head()

columnas = ["Autovaloración", "Autoexperiencia","PrexiónExterna","ExperienciaPersonal","Compañeros","Profesor","Positivos","Negativos"]
tabla2 = tabla2_raw[columnas]

#Estandarizado 
st = StandardScaler()
tabla2_st = st.fit_transform(tabla2)
tabla2_st = pd.DataFrame(tabla2_st, columns=columnas)


# BEST MODEL
agglo_model = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo_model.fit_predict(tabla2_st)


# Assuming you have your dataset with features and labels
# X contains the features and y contains the labels
# Replace X and y with your actual dataset
X = tabla2_st  # Example data with 100 samples and 10 features
y = agglo_labels # Example labels (0, 1, or 2) for 100 samples

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=8, activation='relu'))  # Input layer with 10 features
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
model.save("ninos_bestmodel.keras")

# StandardScaler

# Save the scaler to a file
dump(st, 'ninos_scaler.joblib')
