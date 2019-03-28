import cv2
import numpy as np
import glob
import math
import os
from rgb_extraction import *
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import splrep, splev

# load model from disk
print("[INFO] loading from model...")
net = cv2.dnn.readNetFromCaffe('C:/Users/Ferhat/Downloads/facedetectionOpenCV-master/facedetectionOpenCV-master/deploy.prototxt.txt', 'C:/Users/Ferhat/Downloads/facedetectionOpenCV-master/facedetectionOpenCV-master/res10_300x300_ssd_iter_140000.caffemodel' )

# folder_path = 'C:/Users/Ferhat/Desktop/HeartRateEst/01-04/01-04/'
folder_path = 'C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Frames/'
roi_folder_path = 'C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/roi_of_samples_CE/'
skin_pixels_only_folder_path = 'C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/roi_w_skinpixels_only_CE/'
total_undetermined_images = 0

def read_samples(folder_path):
    # load the input image and construct an input blob for the image and resize image to
    # fixed 300x300 pixels and then normalize it
    images = [cv2.imread(file) for file in glob.glob(folder_path + '*png')]
    return images
def read_samples2(folder_path):
    images = [Image.open(file, 'r') for file in glob.glob(folder_path + '*png')]
    return images

def find_roi(image):
    # mode 1: Return image with box
    # mode 2: Return original image with box coordinates
    original_image = image

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    face_count = 0
    roi_found_count = 0
    d = 1
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            roi_found_count +=1
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            break
            
        if roi_found_count == 0:
            print("Could not determine a roi for this image!")
            total_undetermined_images += 1


    return original_image, startX, startY, endX, endY 
    

def main():

    samples = read_samples(folder_path)
    d = 1
    img_save_folder_path = ''
    for sample in samples:
        image_w_roi, startX, startY, endX, endY  = find_roi(sample)
        filename = "C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Samples_w_roi_CE/sample_w_roi_%d.png" %d
        cv2.imwrite(filename, image_w_roi)

        
        roi = image_w_roi[startY + 2 :endY - 2 , startX + 2:endX - 2] 
        filename = "C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/roi_of_samples_CE/roi_of_sample%d.png" %d
        cv2.imwrite(filename, roi)

        d+=1

    print("ROI's for: " + str(len(samples) - total_undetermined_images) + " images are determined!")

def main2():
    # RGB extraction
    samples = read_samples(skin_pixels_only_folder_path)
    
    samples_avg_red_list = []
    samples_avg_green_list = []
    samples_avg_blue_list = []
    avg_red_window = []
    avg_blue_window = []
    avg_green_window = []
    d = 1

    for sample in samples:  
        avg_red, avg_green, avg_blue, roi_w_skinpixels_only = get_avg_RGB(sample)

        # filename = "C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/roi_w_skinpixels_only_1-4/sample_w_roi_%d.png" %d
        # roi_w_skinpixels_only = np.array(roi_w_skinpixels_only)
        # cv2.imwrite(filename, roi_w_skinpixels_only)
        d += 1

        samples_avg_red_list.append(avg_red)
        samples_avg_green_list.append(avg_green)
        samples_avg_blue_list.append(avg_blue) 
        print("Frame process done %d" %d )
        
    print(len(samples_avg_red_list))
    print(len(samples_avg_green_list))
    print(len(samples_avg_blue_list))

    

    f= open("C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Signals_CE/samples_avg_red_list.txt","w+")

    for item in samples_avg_red_list:
        f.write("%s\n" % item)

    f.close()

    f= open("C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Signals_CE/samples_avg_green_list.txt","w+")

    for item in samples_avg_green_list:
        f.write("%s\n" % item)

    f.close()


    f= open("C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Signals_CE/samples_avg_blue_list.txt","w+")

    for item in samples_avg_blue_list:
        f.write("%s\n" % item)

    f.close()



    '''


    for window_index in range(0, len(samples)-899):
        sum_red_window = 0
        sum_green_window = 0
        sum_blue_window = 0
        for x in range(900):
            sum_red_window += samples_avg_red_list[window_index + x]
            sum_green_window += samples_avg_green_list[window_index + x]
            sum_blue_window += samples_avg_blue_list[window_index + x]
        avg_red_window.append(sum_red_window/900)
        avg_green_window.append(sum_green_window/900)
        avg_blue_window.append(sum_blue_window/900)
        

    print(len(avg_red_window))
    print(len(avg_green_window))
    print(len(avg_blue_window))
    
    '''


    #smooth_green_list = smooth(samples_avg_green_list)
    x = np.arange(0, len(samples_avg_blue_list))
    
    plt.figure(1)
    plt.subplot(3,1,1)
    
    plt.plot(x, samples_avg_red_list, 'r')
    plt.xlabel('Time')
    plt.ylabel('Avg Red')
    plt.title('Red Signal')

    plt.subplot(3,1,2)
    plt.plot(x, samples_avg_green_list, 'g')
    plt.xlabel('Time')
    plt.ylabel('Avg Green')
    plt.title('Green Signal')

    plt.subplot(3,1,3)
    plt.plot(x, samples_avg_blue_list, 'b')
    plt.xlabel('Time')
    plt.ylabel('Avg Blue')
    plt.title('Blue Signal')

    plt.subplots_adjust(hspace= 0.9)
    plt.show()


def test_main():
    # RGB extraction
    samples = read_samples2(roi_folder_path)
        
    red = []
    green = []
    blue = []  
    result =[]
    for image in samples:
        width, height = image.size

        pixel_values = list(image.getdata())  
        pixel_values = np.array(pixel_values).reshape((width, height, 3))
        
        total=0
        for x in range(width):
            average = pixel_values[x]
            for y in range(height):
                mid = average[y]
                if mid [0] > 95 and mid [1] > 40 and mid [2] > 20 and (max(mid[0],mid[1],mid[2])-min(mid[0],mid[1],mid[2])) >15 and math.fabs(mid[0]-mid[1])>15 and mid[0]>mid[1] and mid[0]>mid[2]:
                    total=mid+total
                else:
                    excep=+1
                    pixel_values[x][y][0]=255
                    pixel_values[x][y][1]=255
                    pixel_values[x][y][2]=255
                    
        ##cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
        
        total=total/((width*height)-excep)
        #print(total)
        #print(numff)
        red.append(total[0])
        green.append(total[1])
        blue.append(total[2])


    #plt.plot( range(0,len(green)),green, 'ro')
    plt.plot(green)
    #plt.axis([0, 1500, 0, 120])
    plt.show()

def save_skin_only_images():
    # RGB extraction
    '''
    samples = read_samples(roi_folder_path)
    avg_red, avg_green, avg_blue, roi_w_skinpixels_only = get_avg_RGB(samples[25])
    print("Avg red is %f" % avg_red) 
    '''
    samples = read_samples(roi_folder_path)
    d = 1
    for sample in samples:
        skin_only_image = skin_detector(sample)
        filename = "C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/roi_w_skinpixels_only_CE/skin_CE_%d.png" %d
        cv2.imwrite(filename, skin_only_image)
        d+=1
        print("[INFO] skin only image saved N:%d..." %d )

def test_main3():
    samples = read_samples(skin_pixels_only_folder_path)
    avg_r_list = []
    avg_g_list = []
    avg_b_list = []
    d = 0

    for sample in samples:  
        avg_r, avg_g, avg_b = calculate_average_RGB(sample)
        avg_r_list.append(avg_r)
        avg_g_list.append(avg_g)
        avg_b_list.append(avg_b)
        d+=1
        print("[INFO] Constructing Graphs:%d..." %d )

    f= open("C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/new_samples_avg_red.txt","w+")

    for item in avg_r_list:
        f.write("%s\n" % item)

    f.close()


    x = np.arange(0, len(avg_g_list))

    plt.figure(1)
    plt.subplot(3,1,1)
    
    plt.plot(x, avg_r_list, 'r')
    plt.xlabel('Time')
    plt.ylabel('Avg Red')
    plt.title('Red Signal')

    plt.subplot(3,1,2)
    plt.plot(x, avg_g_list, 'g')
    plt.xlabel('Time')
    plt.ylabel('Avg Green')
    plt.title('Green Signal')

    plt.subplot(3,1,3)
    plt.plot(x, avg_b_list, 'b')
    plt.xlabel('Time')
    plt.ylabel('Avg Blue')
    plt.title('Blue Signal')

    plt.subplots_adjust(hspace= 0.9)
    plt.show()


def plot_data():
    avg_r_list = []
    text_file = open("C:/Users/Ferhat/Desktop/Project - Heart Rate Estimation/Signals_CE/samples_avg_green_list.txt", "r")
    lines = text_file.readlines()
    lines = [float(x) for x in lines]
    lines = smooth(lines)
    # lines = smooth2(lines)
    d = 0
    text_file.close()

    x = np.arange(0, len(lines))
    
    plt.figure(1)
    plt.subplot(3,1,1)
    
    plt.plot(x, lines, 'r')
    plt.xlabel('Time')
    plt.ylabel('Avg Red')
    plt.title('Red Signal')

    plt.show()


if __name__== "__main__":
    # main()
    # main2()
    # test_main2()
    # save_skin_only_images()
    # test_main3()
    plot_data()
    # video_to_frames()
