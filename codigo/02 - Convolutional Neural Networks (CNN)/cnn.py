# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:04:47 2021

@author: msantamaria
"""

# Parte 1 - Construir el modelo de CNN

# Importar las librerías y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.models import load_model
import os

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
        
input_size = (128, 128)
batch_size = 32   
    
# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (*input_size,3), activation = 'relu'))

# Paso 2 - Max Pooling
classifier.add(MaxPool2D(pool_size = (2,2))) # 2x2 is optimal

# Agregado de otra capa de convolución y max pooling
classifier.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2,2)))

# Agregado de otra capa de convolución y max pooling
classifier.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
classifier.add(MaxPool2D(pool_size = (2,2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dropout(rate = 0.5))

# Capa de salida
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilar la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Parte 2 - Ajustar la CNN a las imágenes para entrenar
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        r'C:\Repositories\deeplearning-az\codigo\02 - Convolutional Neural Networks (CNN)\dataset\training_set',
        target_size=input_size,
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        r'C:\Repositories\deeplearning-az\codigo\02 - Convolutional Neural Networks (CNN)\dataset\test_set',
        target_size=input_size,
        batch_size=batch_size,
        class_mode='binary')

# Create a loss history
history = LossHistory()

classifier.fit(
        train_generator,
        steps_per_epoch = train_generator.n // batch_size,
        epochs = 90,
        workers = 12,
        max_queue_size = 100,
        validation_data = validation_generator,
        validation_steps = validation_generator.n // batch_size)

print("The model class indices are:", train_generator.class_indices)

# Save model
model_backup_path = os.path.join(r'C:\Repositories\deeplearning-az\codigo\02 - Convolutional Neural Networks (CNN)\cat_or_dogs_model.h5')
classifier.save(model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(r'C:\Repositories\deeplearning-az\codigo\02 - Convolutional Neural Networks (CNN)\loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()

classifier = load_model(r'C:\Repositories\deeplearning-az\codigo\02 - Convolutional Neural Networks (CNN)\cat_or_dogs_model.h5')

# Loss value & metrics values
print(classifier.evaluate(train_generator))
print(classifier.evaluate(validation_generator))

# For a single prediction
import numpy as np
from keras.preprocessing import image
 
test_image = image.load_img(r"C:\Repositories\deeplearning-az\codigo\02 - Convolutional Neural Networks (CNN)\dataset\single_prediction\cat_or_dog_8.jpg",
                            target_size = input_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_on_batch(test_image)

print(result[0][0])

if result[0][0] > 0.5: 
    print("perro")
else:
    print("gato")
