# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:01:51 2021

@author: msantamaria
"""

# Regresión Lógistica

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

# Tratamiento de los NAs
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values = np.nan,
#                         strategy = "mean")
# imputer = imputer.fit(X[:,1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])
# X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Codificar datos categóricos
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelEncoder_X = LabelEncoder()
# X[:,0] = labelEncoder_X.fit_transform(X[:,0])
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], # The column numbers to be transformed (here is [0] but can be [0, 1, 3])    
#     remainder = 'passthrough' # Leave the rest of the columns untouched
# )
# X = np.array(ct.fit_transform(X), dtype = np.float)
# labelEncoder_y = LabelEncoder()
# y = labelEncoder_y.fit_transform(y)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el modelo de Regresión Logística en el Conjunto de Entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
#classifier = LogisticRegression(solver = "liblinear", multi_class = "ovr", n_jobs = 1, random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# For a single observation
#y_pred = classifier.predict(sc_X.transform(np.array([[20, 50000]])))

# How to re-transform Age and Salary back to their original scale?
#X_train = sc_X.inverse_transform(X_train)
#X_test = sc_X.inverse_transform(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()
