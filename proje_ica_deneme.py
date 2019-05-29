# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:18:36 2019

@author: SADO
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:36:19 2019

@author: SADO
"""#$ python proje_ica_deneme.py --shape-predictor shape_predictor_5_face_landmarks.dat
from PIL import Image
import cv2
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import numpy
import math
from imutils.video import VideoStream
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy.signal import welch
from scipy import signal

numff=0 
red = []
green = []
blue = []
result =[]
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args()) 
# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
 

video_capture = cv2.VideoCapture("vid.avi")
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

while True:
    ret, frame = video_capture.read()
    if ret:
        numff +=1
        frame = cv2.flip(frame, 1)  
        #frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)          #  !!! cigdem hocanın videosu icin 
        #frame = imutils.resize(frame, height=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)         
				
   # loop over the face detections
        for rect in rects:
            width = None
            height = None
            pixel_values = None
            image = None
            bX= None
            bY=None
            bW=None
            bH=None
            excep=0
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            bH=bW
            crop_img = frame[bY:(bY+bH), bX:(bX+bW)]
            cv2.imwrite("frame.jpg", crop_img) 
            image = Image.open('frame.jpg', 'r')      
            width, height = image.size
            pixel_values = list(image.getdata())  
            pixel_values = numpy.array(pixel_values).reshape((width, height, 3))
            #cv2.imwrite("face%d.jpg" % numff, pixel_values)
            total=None
            total=0
            width=round((width/5)*3)
            for x in range(width):
                average = pixel_values[x]
                for y in range(height):
                    mid = average[y]
                    a=mid[2]
                    mid[2]=mid[0]
                    mid[0]=a
                    if (mid [2] > 95 and mid [1] > 40 and mid [0] > 20 and (max(mid[0],mid[1],mid[2])-min(mid[0],mid[1],mid[2])) >15 and math.fabs(mid[2]-mid[1])>15 and mid[2]>mid[1] and mid[2]>mid[1]):
                        total=mid+total
                    else:
                        excep+=1
                        pixel_values[x][y][0]=255
                        pixel_values[x][y][1]=255
                        pixel_values[x][y][2]=255
                        
            #cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
            #cv2.imwrite("renk%d.jpg" % numff, pixel_values)
            total=total/((width*height)-excep)
    

            red.append(total[0])
            green.append(total[1])
            blue.append(total[2])
            
            """sumRed=0
            sumGreen=0
            sumBlue=0
            if numff>900:
                for n in range(1,901):
                    sumRed=sumRed+red[len(red)-n]
                    sumGreen=sumGreen+green[len(green)-n]
                    sumBlue=sumBlue+blue[len(blue)-n]
                sumRed=sumRed/900
                sumGreen=sumGreen/900
                sumBlue=sumBlue/900
                if numff%30==0:
                    print(sumRed)
                    print(sumGreen)
                    print(sumBlue)
                    result.append(sumGreen)
                """       
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            for (i, (x, y)) in enumerate(shape):
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video_capture.release()
cv2.destroyAllWindows()
#ica
for x in range(len(red)-900): 
    plt.clf()
    S=None
    S = numpy.c_[red[x:900+x],blue[x:900+x],green[x:900+x]]
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(S)
    
    models = [ S_]
    names = ['ICA recovered signals']
    colors = ['red', 'steelblue', 'orange']

    deneme=[]
    for sig, color in zip(models[0].T, colors):
        deneme.append(sig)
    i=1
    first_comp=deneme[1]
    freqs, psd = signal.welch(first_comp)
    plt.plot(psd)
    plt.title('PSD: power spectral density')
    plt.xlabel('Frequency')
    plt.draw()
    plt.pause(0.000001)

#A_ = ica.mixing_
#inverse_transform(X[, copy])	Transform the sources back to the mixed data (apply mixing matrix).
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

#assert numpy.allclose(X, numpy.dot(S_, A_.T) + ica.mean_)
"""S = numpy.c_[red,blue,green]
ica = FastICA(n_components=3)
S_ = ica.fit_transform(S)


models = [ S_]
names = ['ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

deneme=[]

for sig, color in zip(models[0].T, colors):
    deneme.append(sig)
i=1
for sig, color in zip(models[0].T, colors):
    plt.subplot(3, 1, i)
    plt.title(names[0])
    plt.plot(sig, color=color)
    i=i+1
plt.show()"""

"""first_comp=deneme[0]
temp=first_comp[0:900]
freqs, psd = signal.welch(temp)
plt.plot(psd)
plt.show()"""
"""plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()
plt.show()"""
























