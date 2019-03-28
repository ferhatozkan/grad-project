from PIL import Image
import glob
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('C:/Users/Ferhat/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

image_list = []
for filename in glob.glob('C:/Users/Ferhat/Desktop/HeartRateEst/05-06/05-06/*.png'): 
    im=Image.open(filename)
    image_list.append(im)
#print(len(image_list))
#image_list[238].show()
img = cv.imread('C:/Users/Ferhat/Desktop/HeartRateEst/05-06/05-06/Image1392651511607816000.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
