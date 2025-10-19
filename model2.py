#import kerastuner as kt
#from tensorflow import keras
import tensorflow as tf
# from kerastuner.tuners import RandomSearch
# from kerastuner.engine.hyperparameters import HyperParameter as hp
# from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten
# from keras.models import Sequential 
# from keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.applications import VGG19
from keras import layers
# from keras.preprocessing import image
# import matplotlib.pyplot as plt
# import seaborn as sns
import keras

def CNN(n):
    model = keras.models.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(3, activation= 'softmax')
    ])
    #compile model
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return model

