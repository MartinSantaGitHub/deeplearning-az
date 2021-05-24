# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:04:07 2021

@author: msantamaria
"""

# Parte 1 - Pre procesado de datos

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Churn_Modelling_Balanced.csv")
X = dataset.iloc[:,4:14].values # iloc -> index localization
y = dataset.iloc[:,-1].values

dataset.isnull().mean()

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto',drop='first'), [1])], # The column numbers to be transformed (here is [0] but can be [0, 1, 3])    
    remainder = 'passthrough' # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype = np.float)

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 0)
 
# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # Se podría utilizar también un MinMaxScaler dado que la capa de salida utiliza una función de activación sigmoide
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

# Matriz de confusión y otras metricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score 
from sklearn.metrics import accuracy_score

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.metrics import Recall

from sklearn.model_selection import GridSearchCV
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import BinaryCrossentropy

es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=0, patience=50)

def build_classifier(optimizer = "adam", loss = 'binary_crossentropy', activation = 'relu'):
    # Inicializar la RNA 
    classifier = Sequential()    
    
    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units = 8, kernel_regularizer = None, kernel_initializer = 'uniform', activation = activation, input_dim = 11))
    
    #classifier.add(Dropout(rate = 0.05))
    
    # Añadir la segunda capa oculta
    classifier.add(Dense(units = 8, kernel_regularizer = None, kernel_initializer = 'uniform', activation = activation))    
    
    #classifier.add(Dropout(rate = 0.05))
    
    #classifier.add(Dense(units = 24, kernel_regularizer = None, kernel_initializer = 'uniform', activation = activation))
    
    # Conviene poner una capa de Dropout al final de todas las capas para
    # resolver los problemas de overfitting
    
    classifier.add(Dropout(rate = 0.10)) # Como mucho llegar a 0.5    
    
    # Añadir la capa de salida
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compilar la RNA
    classifier.compile(optimizer = optimizer, loss = loss, metrics = [Recall()])
    
    # Devolver el clasificador
    return classifier

classifier = build_classifier(optimizer = "adam")

# Ajustamos la RNA al Conjunto de Entrenamiento
history = classifier.fit(X_train, y_train, validation_split = 0.2, batch_size = 4, epochs = 64)

plt.plot(history.history['recall_7'])
plt.plot(history.history['val_recall_7'])
plt.title('Model Recall')
plt.ylabel('Recall')
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

# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

cm = confusion_matrix(y_test, y_pred)

sensitivity = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(sensitivity)
print(accuracy)

single_pred = classifier.predict(sc_X.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
single_pred = single_pred > 0.5

print(single_pred)

# Parte 4 - Evaluar, mejorar y Ajustar la RNA

# Evaluar la RNA

#X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size = 0.20, random_state = 0)

classifier_cv = KerasClassifier(build_fn = build_classifier, validation_split = 0.2, batch_size = 4, nb_epoch = 64)
accuracies = cross_val_score(estimator = classifier_cv, X = X_train, y = y_train, cv = 10, n_jobs = -1, scoring = "recall")

mean_cv = accuracies.mean()
variance_cv = accuracies.std()

print(mean_cv)
print(variance_cv)

# Mejorar la RNA

# Regularización de Dropout para evitar el overfitting

# Ver el agregado de la capa Dropout tanto en la parte 2 como en la parte 3 cuando
# se construye la RNA

# Ajustar la RNA

def build_classifier_1(optimizer = "adam", 
                       initializer = "uniform",
                       regularizer = None,
                       b_initializer = "zeros"):
    
    classifier = Sequential()
        
    classifier.add(Dense(units = 6, 
                         use_bias = True,
                         kernel_initializer = initializer,
                         kernel_regularizer = regularizer,
                         bias_initializer = b_initializer,
                         activation = 'selu', input_dim = 11))    
    
    #classifier.add(Dropout(rate = 0.1))
    
    #classifier.add(LeakyReLU())
    
    classifier.add(Dense(units = 6, 
                         use_bias = True,
                         kernel_initializer = initializer,
                         kernel_regularizer = regularizer,
                         bias_initializer = b_initializer,
                         activation = 'selu'))
        
    classifier.add(Dense(units = 6, 
                         use_bias = True,
                         kernel_initializer = initializer,
                         kernel_regularizer = regularizer,
                         bias_initializer = b_initializer,
                         activation = 'selu'))
    
    #classifier.add(Dropout(rate = 0.1))
    
    #classifier.add(LeakyReLU())
    
    classifier.add(Dropout(rate = 0.21))
       
    classifier.add(Dense(units = 1, kernel_initializer = initializer, activation = 'tanh'))
       
    classifier.compile(optimizer = optimizer, loss = 'hinge', metrics = ['accuracy'])
        
    return classifier

# Optimizadores más eficientes en los casos de probabilidades, 'adam' y'rmsprop'.

def build_classifier_2(optimizer = "adam"):
    # Inicializar la RNA 
    classifier = Sequential()
    
    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units = 8, kernel_regularizer = None, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    classifier.add(Dropout(rate = 0.05))
    
    # Añadir la segunda capa oculta
    classifier.add(Dense(units = 8, kernel_regularizer = None, kernel_initializer = 'uniform', activation = 'relu'))
    
    classifier.add(Dropout(rate = 0.05))
    
    classifier.add(Dense(units = 8, kernel_regularizer = None, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Conviene poner una capa de Dropout al final de todas las capas para
    # resolver los problemas de overfitting
    
    classifier.add(Dropout(rate = 0.05)) # Como mucho llegar a 0.5
    
    # Añadir la capa de salida
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compilar la RNA
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Devolver el clasificador
    return classifier

classifier_gs = KerasClassifier(build_fn = build_classifier, validation_split = 0.2)

parameters = {
    'batch_size': [4,8,16],    
    'nb_epoch' : [50,64,72],
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier_gs, 
                           param_grid = parameters,
                           scoring = 'recall', 
                           cv = 10, 
                           n_jobs= -1)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)

# Puesta a prueba
classifier = Sequential()
        
classifier.add(Dense(input_dim = 11,
                     units = 6,                     
                     kernel_initializer = "uniform",                     
                     activation = 'selu'))    
    
classifier.add(Dense(units = 6,                      
                     kernel_initializer = "uniform",                     
                     activation = 'selu'))
        
classifier.add(Dense(units = 6,                      
                     kernel_initializer = "uniform",                         
                     activation = 'selu'))
        
classifier.add(Dropout(rate = 0.21))
       
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'tanh'))
       
classifier.compile(optimizer = "adam", loss = 'hinge', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 3, epochs = 170)

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

cm = confusion_matrix(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(sensitivity)
print(accuracy)

# Puesta a prueba CV
def build_classifier_2():
    classifier = Sequential()
        
    classifier.add(Dense(input_dim = 11,
                         units = 6,                     
                         kernel_initializer = "uniform",                     
                         activation = 'selu'))    
    
    classifier.add(Dense(units = 6,                      
                         kernel_initializer = "uniform",                     
                         activation = 'selu'))
        
    classifier.add(Dense(units = 6,                      
                         kernel_initializer = "uniform",                         
                         activation = 'selu'))
        
    classifier.add(Dropout(rate = 0.21))
       
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = 'tanh'))
       
    classifier.compile(optimizer = "adam", loss = 'hinge', metrics = ['accuracy'])
    
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier_2, batch_size = 3, nb_epoch = 170)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

print(mean)
print(variance)
