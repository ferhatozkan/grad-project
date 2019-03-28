print('hello')
import glob
import cv2
import os


def read_samples(folder_path):
    # load the input image and construct an input blob for the image and resize image to
    # fixed 300x300 pixels and then normalize it
    images = [cv2.imread(file) for file in glob.glob(folder_path + '*png')]
    return images

folder_path = 'C:/Users/Ferhat/Desktop/HeartRateEst/02-05/02-05/'
samples = read_samples(folder_path)

height, width, channels = samples[0].shape
# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*"XVID"), 15.0, (width, height))

for frame in samples:

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
print('yay')
cv2.destroyAllWindows()
