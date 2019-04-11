"""Convolutional Neural Network for Fashion MNIST Classification.
Team #name
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import cv2
import imutils
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from pnslib import utils

#at first import desired image
image = cv2.imread("testpicture.jpg")

#image rescalling, greyscale and edgemap and reducing noise with gaussian blur
image = imutils.resize(image,height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)


#cv2.imshow("testpicture", edged)
#cv2.waitKey(0)
#cv2.destroyWindow("testpicture")

#thresholds the picture
thresh = cv2.threshold(edged, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#invert image to find contours
imagem = cv2.bitwise_not(thresh)

#gives back the contours
cnts = cv2.findContours(imagem, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

print (cnts[0][0].shape)
contours = cnts[0][0]

#draws a rectangle around the number
x,y,w,h = cv2.boundingRect(contours)

im2 = cv2.rectangle(imagem, (x,y) , (x+w,y+h) ,(0,255,0),2)
im3 = cv2.rectangle(image, (x,y) , (x+w,y+h) ,(0,255,0),2)
#cv2.imshow("im2",im2)
#cv2.waitKey(0)
#cv2.destroyWindow("im2")

#cropping to quadratic window, with sidelength h
crop_img = im2[y-100:y+h+100, x-100:x+h+100]
crop_img2 = im3[y-100:y+h+100, x-100:x+h+100]
cv2.imshow("cropped", crop_img2)
cv2.waitKey(0)

#now resize it
resimage = cv2.resize(crop_img,(28,28), interpolation = cv2.INTER_CUBIC)

newimage1 = np.expand_dims(resimage,axis = 2)
newimage2 = np.expand_dims(newimage1,axis = 0)

print(newimage2.shape)

#show resized image
cv2.imshow("rescaled",resimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

#load trained moedel
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('trained_number_model.hdf5')

#models.predict takes 4 dim array with 10 images
#prediction = model.predict(newimage2)
preds = np.argmax(model.predict(newimage2), axis=1).astype(np.int)
labels = ["0", "1", "2", "3", "4","5","6","7","8","9"]
print("It should be a: ")
print( labels[preds[0]])


