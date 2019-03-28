from PIL import Image
import cv2
import numpy as np
import argparse
import imutils
'''
This code gets an image and returns averaged RGB values in image

pix is a array with [ B G R] ordered of RGB values
'''


# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


def get_avg_RGB(image):
    #im = Image.open('C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/red-AA0114-HexColorRGB-170-1-20.jpg') # Can be many different formats.
    #image = cv2.imread('C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/dog.jpg')
    im = Image.fromarray(image, 'RGB')
    pix = im.load()
    # print(im.size)  # Get the width and hight of the image for iterating over
    total_red = 0
    total_green = 0
    total_blue = 0
    total_number_of_pixels = im.size[0] * im.size[1]
    number_of_skin_pixel = 0

    # Create a New empty image. We will put the skin pixels on this image
    w, h = im.size[0], im.size[1]
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            data[x, y] = [255,255,255]
    roi_w_skinpixels_only = Image.fromarray(data, 'RGB')
    
    for x in range(0, im.size[0]):
        for y in range(0, im.size[1]):
            # Skin detection formula
            red_pixel = pix[x,y][2]
            green_pixel = pix[x,y][1]
            blue_pixel = pix[x,y][0]
            if(red_pixel == 255):
                continue
            total_blue += pix[x,y][0]
            total_green += pix[x,y][1]
            total_red += pix[x,y][2]
            number_of_skin_pixel += 1
            

            '''
            if( red_pixel > 95 and blue_pixel > 20 and max(red_pixel, green_pixel, blue_pixel) - min(red_pixel, green_pixel, blue_pixel) > 15 and 
            abs(red_pixel - green_pixel) > 15 and  red_pixel > green_pixel and  red_pixel > blue_pixel):
                total_blue += pix[x,y][0]
                total_green += pix[x,y][1]
                total_red += pix[x,y][2]
                number_of_skin_pixel += 1
                # Add skin pixel to white image
                current_pixel = im.getpixel( (x,y) )
                roi_w_skinpixels_only.putpixel( (x, y) , current_pixel )     
            '''  

    avg_red = total_red / float(number_of_skin_pixel)
    avg_green = total_green / float(number_of_skin_pixel)
    avg_blue = total_blue / float(number_of_skin_pixel)
    
    # print(number_of_skin_pixel)
    # print(im.size[0] * im.size[1])

    # print(int(avg_red))
    # print(int(avg_green))
    # print(int(avg_blue))
    
    return avg_red, avg_green, avg_blue, roi_w_skinpixels_only


def smooth(dataset):
    dataset_length = len(dataset)
    th = 0.90
    for x in range(0, dataset_length):
        if(x < 3):
            dif1 = dataset[x+1] - dataset[x]
            dif2 = dataset[x+2] - dataset[x]
            dif3 = dataset[x+3] - dataset[x]
            if(abs(dif1) > th and abs(dif2) > th and abs(dif3) > th):
                dataset[x] = dataset[x+1] 
        elif(x > dataset_length-4):
            dif1 = dataset[x] - dataset[x-1]
            dif2 = dataset[x] - dataset[x-2]
            dif3 = dataset[x] - dataset[x-3]
            if(abs(dif1) > th and abs(dif2) > th and abs(dif3) > th):
                dataset[x] = dataset[x-1] 
        else:
            dif1 = dataset[x+1] - dataset[x]
            dif2 = dataset[x+2] - dataset[x]
            dif3 = dataset[x+3] - dataset[x]
            dif4 = dataset[x] - dataset[x-1]
            dif5 = dataset[x] - dataset[x-2]
            dif6 = dataset[x] - dataset[x-3]
            if(abs(dif1) > th and abs(dif2) > th and abs(dif3) > th and abs(dif4) > th and abs(dif5) > th and abs(dif6) > th):
                dataset[x] = ( dataset[x-1] + dataset[x+1] ) / 2.0 
    return dataset

def smooth2(dataset):
    dataset_length = len(dataset)
    th = 1.20
    elements = np.array(dataset)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    for x in range(0, dataset_length):
        if(x < 2):
            if (x < mean - 2 * sd or x > mean + 2 * sd):
                dataset[x] = (dataset[x+1] + dataset[x+2]  + dataset[x+3]) / 2.0
        elif(x > dataset_length-3):
            if (x < mean - 2 * sd or x > mean + 2 * sd):
                dataset[x] = (dataset[x-1] + dataset[x-2]) / 2.0
        else:
            if (x < mean - 2 * sd or x > mean + 2 * sd):
                dataset[x] = (dataset[x-1] + dataset[x+1]) / 2.0
    return dataset



def skin_detector(frame):
    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    h = skin.shape[0]
    w = skin.shape[1]
    for x in range(0, h):
        for y in range(0, w):
            pixel = skin[x,y]
            if(pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0):
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
            

    return skin

def calculate_average_RGB(image):

    total_red = 0
    total_green = 0
    total_blue = 0
    number_of_skin_pixel = 0 
    h = image.shape[0]
    w = image.shape[1]
    for x in range(0, h):
        for y in range(0, w):
            pixel = image[x,y]
            if(pixel[0] != 255):
                total_blue += pixel[0]
                total_green += pixel[1]
                total_red += pixel[2]
                number_of_skin_pixel += 1

    avg_red = total_red / float(number_of_skin_pixel)
    avg_green = total_green / float(number_of_skin_pixel)
    avg_blue = total_blue / float(number_of_skin_pixel)
            

    return avg_red, avg_green, avg_blue

def video_to_frames():
    vidcap = cv2.VideoCapture('C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/sabit1.mov')
    vidcap.set(cv2.CAP_PROP_FPS, 60)
    success,image = vidcap.read()
    count = 0
    while success:
        num_rows, num_cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -90, 1)
        img_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
        cv2.imwrite("C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Frames/frame%d.png" % count, img_rotation)        
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1