"""
@author: Ashutosh Agrahari
@Time: 07 Apr 2020 01:25 PM UTC+5:30
"""
import numpy as np
import pandas as pd
import glob
import PIL
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2
from skimage import io
from sklearn.cluster import KMeans
from sklearn.utils import shuffle 
from scipy.ndimage import median_filter, gaussian_filter
import scipy.ndimage
import sys

def resize(filename, size = (256,192)):
    im = Image.open(filename)
    im_resized = im.resize(size, Image.ANTIALIAS)
    return (im_resized)

img_num = 2
fname = 'IMD%03d'%(img_num) + '.bmp'

frame = resize('Data/frames/' + fname)
frame = np.array(frame)

orig = np.array(frame)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray, plt.cm.binary_r)

med_img = median_filter(gray, 1)
#plt.imshow(med_img, plt.cm.binary_r)

edges_inv = cv2.bitwise_not(med_img)
ret,thresh = cv2.threshold(edges_inv,127,255,0)
#plt.imshow(thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img_contours = cv2.drawContours(frame, contours, -1, (0,255,0), 2)

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(orig)
ax1.set_title('Original Image')

ax2.imshow(img_contours)
ax2.set_title('Segmented Image')

plt.savefig('ML_Segmenter_preds.pdf')