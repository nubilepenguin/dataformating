import os
import glob
import cv2

for data in glob.glob('dataset' + '/*.jpg'):
    img = cv2.imread(data)
    print(data)
    print(img.shape)