"""
Spyder Editor
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
diabetesdata = pd.read_csv(r'C:\Users\Usuario\Downloads\diabetes_data_upload.csv')
print(diabetesdata.head())
print ("************************************************************************")

diabetesdata.rename(columns={'class': 'Class',
                           'Class': 'class'}, inplace=True)
print(diabetesdata.head())

# Assign data from first four columns to X variable
X = diabetesdata.iloc[:, 0:1]
print(X.head())
# Assign data from first fifth columns to y variable
y = diabetesdata.select_dtypes(include=[object])
print(y.head())

y.Class.unique()

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


scaler = StandardScaler()
scaler.fit(X_train)


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=100)
print(X.shape)
y.shape
print(y.shape)

"""
mlp.fit(X_train, y_train.values.ravel())


predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
"""