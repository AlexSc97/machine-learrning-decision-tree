from utils import load_data, processed_data, split_data, save_model, graphics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import tree
import matplotlib.pyplot as plt

import pickle

# Cargar datos
url = "https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv"
df = load_data(url)

# Preparar datos
data_processed = processed_data(df)

# Dividir datos
X_train, X_test, y_train, y_test = split_data(data_processed, 'Outcome')

# Inicializar y entrenar el modelo
print(f"Entrenando modelo de arboles de decisiones")
model_decision_tree = DecisionTreeClassifier(random_state=42)
model_decision_tree.fit(X_train, y_train)
print("Modelo entrenado correctamente")
# Realizar predicciones
y_pred_tree = model_decision_tree.predict(X_test)

# Evaluar el modelo
print(f"Evaluando modelo de arboles de decisiones")
accuracy = accuracy_score(y_test, y_pred_tree)
mse = mean_squared_error(y_test, y_pred_tree)
print(f"Modedlo evaluado correctamente, acuaracy y mse")
print(f"Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")

# Guardar modelo

# Graficos
graphics(data_processed)
