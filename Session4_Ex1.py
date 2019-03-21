"""Convolutional Neural Network for Fashion MNIST Classification.

Team fiver
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pnslib import utils

# Load all the ten classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load(
    data_type="full", flatten=False)

num_classes = 10

print("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
train_x = train_x.astype("float32")/255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32")/255.
test_x -= mean_train_x

print("[MESSAGE] Dataset is preprocessed.")

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

# define a model
image = Input(train_x.shape[1:], name="input_layer")
x = Conv2D(filters=20,          # this layer has 20 filters
           kernel_size=(7, 7),  # the filter size is 7x7
           strides=(2, 2),      # horizontal stride is 2, vertical stride is 2
           padding="same")(image)   # pad 0s so that the output has the same shape as input
x = Activation("relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(filters=25,          # this layer has 25 filters
           kernel_size=(5, 5),  # the filter size is 5x5
           strides=(2, 2),      # horizontal stride is 2, vertical stride is 2
           padding="same")(x)   # pad 0s so that the output has the same shape as input
x = Activation("relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(200)(x)				# linear transform to space with 200 units
x = Activation("relu")(x)
x = Dense(10)(x)  				# the second layer has 10 neuron
out = Activation("softmax")(x)  # the activation function is softmax
model = Model(image,out)



print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

# compile the model aganist the categorical cross entropy loss
# and use SGD optimizer, you can try to use different
# optimizers if you want
# see https://keras.io/losses/
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])



print("[MESSAGE] Model is compiled.")

# train the model with fit function
# See https://keras.io/models/model/ for usage
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True,
    vertical_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_x)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_x, train_Y, batch_size=32),
                    steps_per_epoch=len(train_x) / 32, epochs=5,
                    validation_data=datagen.flow(test_x,test_Y))

print("[MESSAGE] Model is trained.")

# save the trained model
model.save("conv-net-fashion-mnist-trained.hdf5")

print("[MESSAGE] Model is saved.")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_x_vis = test_x[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = np.argmax(model.predict(test_x_vis), axis=1).astype(np.int)

labels = ["Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle Boot"]

plt.figure()
for i in range(2):
    for j in range(5):
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(test_x[i*5+j].reshape(28,28), cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i*5+j]],
                   labels[preds[i*5+j]]))
plt.show()