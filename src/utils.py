import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

# Funciones para cargar datos y preparar datos

def load_data(url):
    """

    :param url: Url con la data .csv
    :return: un dataframe de pandas
    """
    print(f"Cargando datos desde la URL: {url}")
    df = pd.read_csv(url)
    print(f"Datos cargados correctamente")
    return df

# Función de limpieza y preparación de datos

def processed_data(df):
    """

    :param df: Dataframe de pandas
    :return: Data frame de pandas limpio y listo para el modelo
    """
    # Realizo una copia del data frame

    data_processed = df.copy()
    print(f"Copia del dataframe realizada correctamente")

    # Busca si hay duplicados y los elimina
    print(f"Buscando duplicados")
    if data_processed.duplicated().sum()> 0:
        data_processed.drop_duplicates(inplace=True)
        print(f"Duplicados eliminados")


    # Busca si hay nulos oh valores 0 en las columnas ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] y los remplaza por nan
    print(f"Remplazando 0 con nan en las columnas seleccionadas")
    cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Reemplazo 0 con np.nan en esas columnas
    data_processed[cols_con_ceros] = data_processed[cols_con_ceros].replace(0, np.nan)

    print("Remplazando valores nan con la media de cada columna")

    # Imputación de nulos con la media de cada columna
    data_processed = data_processed.fillna(data_processed.mean())
    print("Valores remplazados correctamente")

    # Ahora nos quedamos con la data limpia para su posterior entrenamiento
    data_model = data_processed[['Glucose', 'BMI', 'Age', 'Pregnancies', 'Outcome']]

    return data_model

def split_data(df, var_obj):
    """

    :param df: DataFrame listo para el modelo
    :param var_obj: Variable objetivo y
    :return: data dividida en entrenamiento y prueba
    """
    # Selecciono la data para X
    X = df.drop('Outcome', axis=1)

    # Selecciono solo la variable objetivo en este caso Outcome para y
    y = df['Outcome']
    # Utilizo train_test_split para dividir la data y entrenarla
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_model(model, path):
    """

    :param model: modelo
    :param path: ruta de guardado
    :return: Guarda el modelo.pkl en la ruta indicada
    """
    return pickle.dump(model, path)

# Funcion para graficas

def graphics(data_processed):
    grafico = pd.plotting.parallel_coordinates(data_processed, 'Outcome', color=('orange', 'cyan'))
    return grafico

