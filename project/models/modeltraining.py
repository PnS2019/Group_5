from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

from pnslib import utils
from pnslib import ml

kernel_sizes = [(5, 5), (3, 3), (3, 3)]
num_kernels = [32, 25]

pool_sizes = [(2, 2), (2, 2), (2, 2)]
pool_strides = [(2, 2), (2, 2), (2, 2)]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/Users/luc/Documents/Coding Projects/Deep Learning P&S/data/actual/train',
        target_size=(45, 45),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        '/Users/luc/Documents/Coding Projects/Deep Learning P&S/data/actual/test',
        target_size=(45, 45),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle = False)

x = Input(shape=(45,45,1))
y = Conv2D(num_kernels[0], kernel_sizes[0], activation='relu')(x)
y = MaxPooling2D(pool_sizes[0], pool_strides[0])(y)
y = Conv2D(num_kernels[1], kernel_sizes[1], activation='relu')(y)
y = MaxPooling2D(pool_sizes[1], pool_strides[1])(y)
y = Conv2D(num_kernels[2], kernel_sizes[2], activation='relu')(y)
y = MaxPooling2D(pool_sizes[2], pool_strides[2])(y)
y = Flatten()(y)
y = Dense(100, activation='relu')(y)
y = Dense(15, activation='softmax')(y)
model = Model(x, y)


model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=800)

model.save("shit_happens.hdf5")

#maybe adding leaky relu instead of relu to line 45, 47, 49 later
