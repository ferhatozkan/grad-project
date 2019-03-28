import cv2 as cv
from rgb_extraction import *
print('hello')
img = cv2.imread('C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/face_test.jpg')
avg_red, avg_green, avg_blue = get_avg_RGB(img)