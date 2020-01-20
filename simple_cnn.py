# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:23:04 2020
SIMPLE IMPLEMENTATION OF A CONVOLUTIONAL NEURAL NETWORK

@author: elvis
"""

# LOAD LIBRARIES
import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist

# LOAD DATA
print('Loading and preprocessing Data')
print('='*50)
(x_train,y_train), (x_test, y_test) = mnist.load_data()

# RESHAPE DATA
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

# ENCODE CATEGORICAL DATA
y_train = to_categorical(y_train, 10)  # 10 represents the number of labels(0-9)
y_test = to_categorical(y_test, 10)

# NORMALISE DATA

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /=255 
x_test /=255

print('Creating Model')
print('='*50)

# CREATE MODEL
model = Sequential()
model.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))
print('Compiling Model')
print('='*50)

# COMPILE MODEL
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# PRINT MODEL STRUCTURE
model.summary()

print('Training model on train data')
print('='*50)

# TRAIN MODEL ON DATA
model.fit(x_train, y_train, epochs = 5, batch_size = 20)

# EVALUATE ON TEST DATA
print('Evaluation Model on Test data')
print('='*50)
evaluation_score = model.evaluate(x_test,y_test, verbose=1)
print('Test Loss:',evaluation_score[0])
print('Test Accuracy:', evaluation_score[1])

# DO SOME PREDICTION
print('Prediction on some images from the test set')
print('='*50)
pred = model.predict(x_test[:2])
for i in range(len(x_test[:2])):
    print(pred[i].argmax())
plt.imshow(pred)

#You can also use this code to do prediction on a single image from the test set
#Change image index number for other predicttions
image_index_to_predict = 97
plt.imshow(x_test[image_index_to_predict].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index_to_predict].reshape(1, 28, 28, 1))
print(pred.argmax())
