import cv2
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import math
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import splrep, splev
from scipy.misc import toimage
import imutils
from scipy.signal import welch
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from scipy import signal
# Normalize time series data
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

folder_name = 'Subject5'
video_path = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/video/CE.avi'
folder_path = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/frames/01-04/'
signal_path = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/signals/'
roi_folder = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/roi/'
pureDL_ground_truth_path = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/frames/01-04.json'
ground_truth = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/ground_truth/5.txt'


#green = np.loadtxt('C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/test2.txt', delimiter=',') 


# Parameters to play with
WINDOW_SECONDS = 30             # [s] Sliding window length
BPM_SAMPLING_PERIOD = 0.5       # [s] Time between heart rate estimations
fps = 30                            
BPM_L = 40; BPM_H = 230         # [bpm] Valid heart rate range
FILTER_STABILIZATION_TIME = 1   # [s] Filter startup transient
CUT_START_SECONDS = 0           # [s] Initial signal period to cut off
FINE_TUNING_FREQ_INCREMENT = 1  # [bpm] Separation between test tones for smoothing
ANIMATION_SPEED_FACTOR = 2      # [] This makes the animation run faster or slower than real time

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

# load model from disk
print("[INFO] loading from model...")
net = cv2.dnn.readNetFromCaffe('C:/Users/Ferhat/Downloads/facedetectionOpenCV-master/facedetectionOpenCV-master/deploy.prototxt.txt', 'C:/Users/Ferhat/Downloads/facedetectionOpenCV-master/facedetectionOpenCV-master/res10_300x300_ssd_iter_140000.caffemodel' )




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

    crop_img = original_image[startY + 2 :endY - 2 , startX + 2:endX - 2] 
    return crop_img

def skin_detector(frame):
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

def get_avg_RGB(image):
    pix = image.load()
    r = 0
    g = 0
    b = 0
    sum_r = 0
    sum_g = 0
    sum_b = 0
    pix_count = 0
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            image_pix = image.getpixel((x,y))
            
            r = image_pix[0]
            g = image_pix[1]
            b = image_pix[2]
            
            if r == 255 and g == 255 and b == 255:
                continue
            else:
                sum_r += r
                sum_g += g
                sum_b += b
                pix_count += 1
            
    r = sum_r / float(pix_count)
    g = sum_g / float(pix_count)
    b = sum_b / float(pix_count) 
    print(r)
    return r,g,b

def get_signals(signal_path):
    r_list = []
    g_list = []
    b_list = []
    lines = []

    text_file = open(signal_path + 'r_list.txt',"r")
    lines = text_file.readlines()
    r_list = [float(x) for x in lines]
    text_file.close()

    text_file = open(signal_path + 'g_list.txt',"r")
    lines = text_file.readlines()
    g_list = [float(x) for x in lines]
    text_file.close()

    text_file = open(signal_path + 'b_list.txt',"r")
    lines = text_file.readlines()
    b_list = [float(x) for x in lines]
    text_file.close()

    return r_list, g_list, b_list

def plot_data():
    r_list = []
    lines = []
    text_file = open('C:/Users/Ferhat/Desktop/HeartRate-Project/Subject1/signals-SP/g_list.txt')

    lines = text_file.readlines()
    lines = [float(x) for x in lines]
    text_file.close()

    x = np.arange(0, len(lines))
    plt.figure(1)
    plt.plot(x, lines, 'g')
    plt.xlabel('Time')
    plt.ylabel('Avg Signal')
    plt.title('Green Signal')
    plt.show()

def save_signals(r_list, g_list, b_list):
    f= open(signal_path + 'r_list.txt',"w+")
    for item in r_list:
        f.write("%s\n" % item)
    f.close()
    f= open(signal_path + 'g_list.txt',"w+")
    for item in g_list:
        f.write("%s\n" % item)
    f.close()
    f= open(signal_path + 'b_list.txt',"w+")
    for item in b_list:
        f.write("%s\n" % item)
    f.close()

def snr(list_psd, freqs):
    a = int(( (2.5 - 0.75) / 0.134 ))
    window_len = int(len(list_psd) / a)
    start = 0
    end = window_len
    max_peak_index = 0
    max_peak_frequency = 0
    max_peak = 0
    for x in range(len(list_psd) - window_len ):
        total_power = sum(list_psd[y] for y in range(x , x + window_len))
        if(max_peak < total_power):
            max_peak = total_power
            max_peak_index = list_psd.index(max(list_psd[x] for x in range(x, x + window_len)))
    max_peak_frequency = freqs[max_peak_index]
    return max_peak_frequency

def pureDL_ground_truth():
    import json
    with open(pureDL_ground_truth_path) as json_file:  
        data = json.load(json_file)
        ground_truth = [ e['Value']['pulseRate'] for e in data['/FullPackage'] ]
    gt_HR = ground_truth[900:]
    # Should match pulseRate to frames first.
    return gt_HR    

def main():
    video_capture = cv2.VideoCapture(video_path)
    time.sleep(2.0) 
    r_list = []
    g_list = []
    b_list = []
    d = 0
    number_of_frames = 0
    while True:
        return_signal, frame = video_capture.read()
        if return_signal:
            number_of_frames +=1
            # frame = cv2.flip(frame, 1)   Neden var ?
            roi = find_roi(frame)
            # Just to be sure all faces detected correctly
            filename = roi_folder + "roi_%d.png" %d
            cv2.imwrite(filename, roi) 

            skin_roi = skin_detector(roi)
            skin_roi_arr = Image.fromarray(skin_roi)
            r,g,b = get_avg_RGB(skin_roi_arr)
            r_list.append(r)
            g_list.append(g)
            b_list.append(b)
            print(d)
            d +=1
        else:
            break
    print(str(len(g_list)) + " number of samples analyzed." )
    save_signals(r_list,g_list,b_list)
    video_capture.release()
    cv2.destroyAllWindows()

def read_samples(folder_path):
    images = [cv2.imread(file) for file in glob.glob(folder_path + '*png')]
    return images

def main2():
    samples = read_samples(folder_path)
    r_list = []
    g_list = []
    b_list = []
    d = 0
    number_of_frames = 0
    for sample in samples:
        roi = find_roi(sample)
        # Just to be sure all faces detected correctly
        filename = roi_folder + "roi_%d.png" %d
        cv2.imwrite(filename, roi) 

        skin_roi = skin_detector(roi)
        skin_roi_arr = Image.fromarray(skin_roi)
        r,g,b = get_avg_RGB(skin_roi_arr)
        r_list.append(r)
        g_list.append(g)
        b_list.append(b)
        print(d)
        d +=1
    print(str(len(g_list)) + " number of samples analyzed." )
    save_signals(r_list,g_list,b_list)

def ica(r_signal, g_signal, b_signal):    
    # Some initializations and precalculations
    num_window_samples = round(WINDOW_SECONDS * fps);
    bpm_sampling_period_samples = round(BPM_SAMPLING_PERIOD * fps);
    # num_bpm_samples = floor((size(y, 2) - num_window_samples) / bpm_sampling_period_samples);
    fcl = BPM_L / 60
    fch = BPM_H / 60
    predicted_hr = []
    for x in range(len(r_signal)-900):
        plt.clf()
        S = None
        S = np.c_[r_signal[x:900+x],g_signal[x:900+x],b_signal[x:900+x]] #column_stack
        ica = FastICA(n_components=3)
        S_ = ica.fit_transform(S) 
        comp_1 = S_.T[0]
        comp_2 = S_.T[1]
        comp_3 = S_.T[2]

        '''
        sig = None   
        sig = g_signal[x:900+x]
        '''

        fs=30
        comp1_freqs, comp1_psd = signal.welch(comp_1,fs, nperseg=900)
        comp2_freqs, comp2_psd = signal.welch(comp_2,fs, nperseg=900)
        comp3_freqs, comp3_psd = signal.welch(comp_3,fs, nperseg=900)
        green_freqs, green_psd = signal.welch(g_signal[x:900+x],fs, nperseg=900)

        # Filter 0.75 - 2.5 Hz
        comp1_list_psd = []
        for x in range(len(comp1_freqs)):
            if comp1_freqs[x] > 0.75 and comp1_freqs[x] < 2.5:
                comp1_list_psd.append(comp1_psd[x])
            else:
                comp1_list_psd.append(0)
        
        # Filter 0.75 - 2.5 Hz
        comp2_list_psd = []
        for x in range(len(comp2_freqs)):
            if comp2_freqs[x] > 0.75 and comp2_freqs[x] < 2.5:
                comp2_list_psd.append(comp2_psd[x])
            else:
                comp2_list_psd.append(0)
        
        # Filter 0.75 - 2.5 Hz
        comp3_list_psd = []
        for x in range(len(comp3_freqs)):
            if comp3_freqs[x] > 0.75 and comp3_freqs[x] < 2.5:
                comp3_list_psd.append(comp3_psd[x])
            else:
                comp3_list_psd.append(0)

        # Filter 0.75 - 2.5 Hz
        green_list_psd = []
        for x in range(len(green_freqs)):
            if green_freqs[x] > 0.75 and green_freqs[x] < 2.5:
                green_list_psd.append(green_psd[x])
            else:
                green_list_psd.append(0)
                

        comp1_power = snr(comp1_list_psd, comp1_freqs)
        comp2_power = snr(comp2_list_psd, comp2_freqs)
        comp3_power = snr(comp3_list_psd, comp3_freqs)
        green_power = snr(green_list_psd, green_freqs)

        max_power = max(comp1_power, comp2_power, comp3_power)
        #max_power = green_power

        predicted_hr.append(max_power*60)
        predicted_hr = smooth_result(predicted_hr)

    return predicted_hr

def predict_hr():
    r_list, g_list, b_list = get_signals(signal_path)
    predicted_hr = ica(r_list, g_list, b_list)
    return predicted_hr

def smooth_result(hr_signal):
    for x in range(1,len(hr_signal)):
        if(abs(hr_signal[x-1] - hr_signal[x]) >= 6):
            if(hr_signal[x-1] > hr_signal[x]):
                hr_signal[x-1] = hr_signal[x]
            if(hr_signal[x] > hr_signal[x-1]):
                hr_signal[x] = hr_signal[x-1] 
    return hr_signal

def detrend_signal(signal):
    signal_detrended = signal.detrend(signal)
    return signal_detrended

def normalize_signal(signal):
    signal = np.asarray(signal)
    signal = signal.reshape((len(signal), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(signal)
    normalized_signal = scaler.transform(signal)
    return normalized_signal

def moving_average(signal, N=5):
    cumsum = np.cumsum(np.insert(signal, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_ground_truth():
    ground_truth_values=[]
    ground_truth_values = np.loadtxt(ground_truth) 
    #gt_trace = ground_truth[0]
    gt_HR = ground_truth_values[1]
    #gt_time = ground_truth[2]
    gt_HR = gt_HR[900:]
    return gt_HR


""" def read_samples(folder_path):
    # load the input image and construct an input blob for the image and resize image to
    # fixed 300x300 pixels and then normalize it
    images = [cv2.imread(file) for file in glob.glob(folder_path + '*png')]
    return images

jpgimages = 'C:/Users/Ferhat/Desktop/HeartRate-Project/' + folder_name + '/roi/'

images = read_samples(jpgimages)
image = images[0]
skin = skin_detector(image)
cv2.imshow( "Both flip", skin )
cv2.waitKey(0)
 
# close the windows
cv2.destroyAllWindows() 
"""

if __name__== "__main__":
    #main()
    #main2()
    #plot_data()
    #get_ground_truth()
    pred_hr = predict_hr()
    ground_truth = get_ground_truth()

    x = np.arange(0, len(pred_hr))
    plt.figure(1)
    plt.plot(x, pred_hr, 'r')
    plt.plot(x, ground_truth, 'g')
    plt.xlabel('Time')
    plt.ylabel('BPM')
    plt.title('Hr prediction')
    plt.show() 
    
    plt.figure(2)
    diff = np.subtract(pred_hr, ground_truth)
    plt.plot(x, diff, 'b')
    
    mean_squared_error = sum(y ** 2 for y in diff) / float(len(diff))
    print(mean_squared_error)
