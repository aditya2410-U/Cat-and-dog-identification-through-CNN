# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

from platform import python_version
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

import os

import pip
#pip.main(['install','seaborn'])

import seaborn as sn

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time
import datetime as dt
from sklearn.metrics import classification_report, confusion_matrix
import multiprocessing as mp
import shutil

import os

from platform import python_version
import warnings

from PIL import Image
import xml.etree.ElementTree as ET


warnings.filterwarnings("ignore")



# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(r'C:\Users\TOSHIBA\Desktop\CNN\CNN research\Part 2 - Convolutional Neural Networks\dataset\training_set',
                                                 
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r'C:\Users\TOSHIBA\Desktop\CNN\CNN research\Part 2 - Convolutional Neural Networks\dataset\test_set',
                                            
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))



# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

img = training_set.filepaths[np.random.random_integers(low=0, high=training_set.samples)]
print(img)
img = mpimg.imread(img)
plt.imshow(img)

test_set.reset()
predictions = cnn.predict_generator(test_set, steps=len(test_set))
y = np.argmax(predictions, axis=1)

print('Classification Report')
cr = classification_report(y_true=test_set.classes, y_pred=y, target_names=test_set.class_indices)
print(cr)
 
print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, y)
df = pd.DataFrame(cm, columns=test_set.class_indices)
plt.figure(figsize=(80,80))
sn.heatmap(df, annot=True)

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)