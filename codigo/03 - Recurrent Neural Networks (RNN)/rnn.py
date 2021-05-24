# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:52:08 2021

@author: msantamaria
"""

# Parte 1 - Preprocesado de los datos

# Importación de las librerías
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Importar el dataset de entrenamiento
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

# Escalado de características
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Crear una estructura de datos con 60 timesteps y 1 salida
X_train = []
y_train = []

timesteps = 60
n_train = len(training_set_scaled)

for i in range(timesteps, n_train):
    X_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Redimensión de los datos
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Parte 2 - Construcción de la RNR

def build_regressor(optimizer = 'Adam'):
    # Inicialización del modelo
    regressor = Sequential()
    
    # Añadir la primera capa de LSTM y la regularización por Dropout
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    # Añadir la segunda capa de LSTM y la regularización por Dropout
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Añadir la tercera capa de LSTM y la regularización por Dropout
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Añadir la cuarta capa de LSTM y la regularización por Dropout 
    # return_sequences = False para no pasar las secuencias hacia atras en el retorno
    regressor.add(LSTM(units = 50, return_sequences = False))
    regressor.add(Dropout(0.2))
    
    # Añadir la capa de salida
    regressor.add(Dense(units = 1))
    
    # Compilar la RNR
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = 'mean_squared_error')         
    
    # Devolver el regresor
    return regressor

# Ajustar la RNR a nuestro conjunto de entrenamiento
regressor =  build_regressor()
history = regressor.fit(X_train, y_train, epochs = 50, batch_size = 16)

# Parte 3 - Ajustar las predicciones y visualizar los resultados

# Obtener el valor real de las acciones de Enero 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_prices = dataset_test.iloc[:,1:2].values

# Predecir las acciones de Enero de 2017 con la RNR
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
n_test = 80

for i in range(timesteps, n_test):
    X_test.append(inputs[i-timesteps:i,0])   

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizar los resultados obtenidos
plt.plot(real_stock_prices, color = 'red', label = 'Precio Real de la Accion de Google')
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio Predicho de la Accion de Google')
plt.title('Prediccion con una RNR del valor de las acciones de Google')
plt.xlabel('Fecha')
plt.xlabel('Precio de la acción de Google')
plt.legend()
plt.show()

rmse = math.sqrt(mean_squared_error(real_stock_prices, predicted_stock_price))
rmse_relative = rmse / (np.max(real_stock_prices) - np.min(real_stock_prices))

print(rmse_relative)

# Grid Search CV

classifier_gs = KerasClassifier(build_fn = build_regressor)

parameters = {
    'batch_size': [16,32,64],    
    'epochs' : [50,100,150],
    'optimizer' : ['Adam', 'RMSprop', 'Nadam']
}

grid_search = GridSearchCV(estimator = classifier_gs, 
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error', 
                           cv = 10, 
                           n_jobs= -1)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)
