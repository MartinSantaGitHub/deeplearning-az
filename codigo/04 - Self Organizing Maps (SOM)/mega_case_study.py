# -*- coding: utf-8 -*-
"""
Created on Fri May 21 20:07:16 2021

@author: msantamaria
"""

# Parte 1 - Identificar los fraudes potenciales con un SOM

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score 
from sklearn.metrics import accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.metrics import Recall
from sklearn.model_selection import GridSearchCV
from keras.losses import BinaryCrossentropy

# Importar el dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Escalado de características
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Entrenar el SOM
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizar los resultados
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], 
         markeredgecolor = colors[y[i]], markerfacecolor = 'None', 
         markersize = 10, markeredgewidth = 2,)
show()

# Encontrar los fraudes
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,4)], mappings[(5,5)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# Parte 2 - Trasladar el modelo de Deep Learning de no supervisado a supervisado

# Crear la matriz de características
customers = dataset.iloc[:,1:-1].values

# Crear la variable dependiente
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Escalado de variables
sc_X = StandardScaler() 
customers = sc_X.fit_transform(customers)

def build_classifier(optimizer = "Adam", loss = 'binary_crossentropy', activation = 'relu'):
    # Inicializar la RNA 
    classifier = Sequential()    
    
    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = activation, input_dim = 14))            
    
    # Añadir la capa de salida
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compilar la RNA
    classifier.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    
    # Devolver el clasificador
    return classifier

classifier = build_classifier()

# Ajustamos la RNA al Conjunto de Entrenamiento
history = classifier.fit(customers, is_fraud, validation_split = 0.2, batch_size = 1, epochs = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Predicción de los resultados de fraude
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()][::-1]
