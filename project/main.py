from __future__ import print_function, absolute_import
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from PIL import Image
from picamera import PiCamera
import time
from time import sleep
import cv2

#taking a picture with PiCamera save the unedited picture
camera = PiCamera()
camera.start_preview()
sleep(3)
camera.capture('/home/pi/Documents/project/data/pictureRaw.jpg')
camera.stop_preview()


#at first import desired image and convert it to a black and white
imagetobw = Image.open("/home/pi/Documents/project/data/pictureRaw.jpg")
thresh = 55 #has to be chosen wisely lower means higher contrast
fn = lambda x : 255 if x > thresh else 0
bwimageIMG = imagetobw.convert('L').point(fn, mode = '1')
bwimageIMG.save("/home/pi/Documents/project/data/pictureBW.jpg")
showBW = cv2.imread("/home/pi/Documents/project/data/pictureBW.jpg", cv2.COLOR_BGR2GRAY)  #if we use the bwimage without reimporting it, it's an object of a class from the Pillow extension
bwimageIM = Image.open("/home/pi/Documents/project/data/pictureBW.jpg")

#basic commands to show picture here the black and white version of the picture
cv2.imshow("black and white",showBW)
cv2.waitKey(0)
cv2.destroyWindow("balck and white")

image = cv2.imread("/home/pi/Documents/project/data/pictureBW.jpg")

#image rescalling, greyscale and edgemap and reducing noise with gaussian blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

#thresholds the picture
thresh = cv2.threshold(edged, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#invert image to find contours
imagem = cv2.bitwise_not(thresh)

#gives back the contours
(_,cnts,_) = cv2.findContours(imagem, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, cnts, -1, (240, 0, 159), 3)
cv2.imshow("contours", image)
cv2.waitKey(0)


#here we sort the contours from Left to right
i = 0
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
(cnts3, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=False))
aList = "";

#we go over all contours
for i in range(len(cnts3)):
    contours = cnts3[i]
    x, y, w, h = cv2.boundingRect(contours)
    area = w*h
    if area > 250: #area value choosen to filter out unwanted noise
        # draws a rectangle around the number or sign
        im2 = cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im3 = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        im2 = cv2.rectangle(imagem, (x,y) , (x+w,y+h) ,(0,255,0),2)
        im3 = cv2.rectangle(image, (x,y) , (x+w,y+h) ,(0,255,0),2)
        
        cv2.imshow("im3",im3)
        cv2.waitKey(0)
        cv2.destroyWindow("im3")
        
        
        # cropping to quadratic window with sidelength h if it's a number ( since there the height ist bigger than width) and w otherwise
        newh = int(1.1 * h)
        neww = int(1.1 * w)
        #creating a rectangular picture with onlz one digit or sign in it
        croppedtorect = bwimageIM.crop((x-int(newh*0.1), y-int(neww*0.1), x+w+(int(0.1*neww)), y+h+(int(0.1*newh))))
        
        #we create a white image and put the digit or sign in it
        size = max(45,newh, neww)
        croppedquad = Image.new('L', (size, size), (255))
        croppedquad.paste(croppedtorect, (int(size/2 - w/2), 0))

        croppedquad.save("/home/pi/Documents/project/data/croppedquad.jpg")
        croppedquadCV = cv2.imread("/home/pi/Documents/project/data/croppedquad.jpg", cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("cropped", croppedquadCV)
        cv2.waitKey(0)
        cv2.destroyWindow("cropped")
        
    
        # now resize it
        resimage = cv2.resize(croppedquadCV, (45, 45), interpolation=cv2.INTER_CUBIC)
        newimage1 = np.expand_dims(resimage, axis=2)
        newimage2 = np.expand_dims(newimage1, axis=0)

        # load trained model
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = load_model('/home/pi/Documents/project/models/31er.hdf5')

        # models.predict takes 4 dim array with 10 images
        # prediction = model.predict(newimage2)
        preds = np.argmax(model.predict(newimage2), axis=1).astype(np.int)
        #labels = ["=", "-", "+", "1", "2", "3","4","5","6","7","8","9","0","div","times"] # for numbers_trained.hdf5
        labels = ["8", "3", "1", "5", "/", "=", "*", "-", "9", "0", "+", "6", "7", "4", "2"]  # for shit_happens.hdf5

        print("It should be a: ")
        print(labels[preds[0]])

        aList = aList + (labels[preds[0]])

print("the resulting computation is")
print(eval(aList))



cv2.destroyAllWindows()
