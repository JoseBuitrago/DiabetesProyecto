"""
PyCharm
Creado sábado, 13 de febrero de 2021

@author: José Alejandro Buitrago Cardenas / Leydi Esperanza Perez Leal
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#Subir dataframe
print ("************************************************************************")

#Read dataset to pandas dataframe
df = pd.read_csv(r'C:\Users\Usuario\Downloads\diabetes_data_upload.csv')
print(df.head())
print ("************************************************************************")

#Crear una función para LabelEncoder
def Encoder(df):
#crear un objeto 'columnsToEncode' que hará una lista de columnas que tienen valores categóricos
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
#recorrer en iteración las columnas de la lista 'columnsToEncode'
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df

columnsToEncode = list(df.select_dtypes(include=['category','object']))
#pasar marco de datos a través de la función
df = Encoder(df)
#ver conjunto de datos
print(df)