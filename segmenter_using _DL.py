"""
@author: Ashutosh Agrahari
@Time: 27 Mar 2020 11:41 PM UTC+5:30

references:
- https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d

- https://www.tensorflow.org/tutorials/images/segmentation
"""
from time import time
start = time()
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
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

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

# Intersection of Union (IOU)
# Equivalent to JaccardIndex
def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return K.mean(K.equal(y_true, K.round(y_pred)))

# Loading model
img_input = Input(shape = (192, 256, 3))
x = Conv2D(16, (5, 5), padding='same', name='conv1',strides= (1,1))(img_input)
x = BatchNormalization(name='bn1')(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', name='conv2')(x)
x = BatchNormalization(name='bn2')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(64, (4, 4), padding='same', name='conv3')(x)
x = BatchNormalization(name='bn3')(x)
x = Activation('relu')(x)
x = Conv2D(64, (4, 4), padding='same', name='conv4')(x)
x = BatchNormalization(name='bn4')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

x = Dropout(0.5)(x)

x = Conv2D(512, (3, 3), padding='same', name='conv5')(x)
x = BatchNormalization(name='bn5')(x)
x = Activation('relu')(x)
x = Dense(1024, activation = 'relu', name='fc1')(x)
x = Dense(1024, activation = 'relu', name='fc2')(x)

# Deconvolution Layers (BatchNorm after non-linear activation)

x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv1')(x)
x = BatchNormalization(name='bn6')(x)
x = Activation('relu')(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv2')(x)
x = BatchNormalization(name='bn7')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv3')(x)
x = BatchNormalization(name='bn8')(x)
x = Activation('relu')(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(1, (3, 3), padding='same', name='deconv4')(x)
x = BatchNormalization(name='bn9')(x)

x = Dropout(0.5)(x)

x = Activation('sigmoid')(x)
pred = Reshape((192,256))(x)

model = Model(inputs=img_input, outputs=pred)
model.load_weights('trained_Lesion_Segmenter.h5')

def enhance(img):
    sub = (model.predict(img.reshape(1,192,256,3))).flatten()
    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def drawContour(m,s,c,RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p:p==c and 255)

    # Find edges of this contour and make into Numpy array
    thisEdges   = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN  = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m

def resize(filename, size = (256,192)):
    im = Image.open(filename)
    im_resized = im.resize(size, Image.ANTIALIAS)
    return (im_resized)

img_num = 2
fname = 'IMD%03d'%(img_num) + '.bmp'

frame = resize('Data/frames/' + fname)
orig_mask = resize('Data/masks/' + fname)

frame = np.array(frame)
orig_mask = np.array(orig_mask)

#img_pred = model.predict(frame.reshape(1,192,256,3))
plt.figure(figsize=(16,3))
plt.subplot(1,4,1)
plt.imshow(frame)
plt.title('Original Image')

plt.subplot(1,4,2)
plt.imshow(orig_mask, plt.cm.binary_r)
plt.title('Ground Truth')

plt.subplot(1,4,3)
img_pred = model.predict(frame.reshape(1,192,256,3)).reshape(192,256)
plt.imshow(img_pred, plt.cm.binary_r)
plt.title('Predicted Output')

plt.subplot(1,4,4)
plt.imshow(enhance(frame).reshape(192,256), plt.cm.binary_r)
plt.title('Enhanced Predicted Output')

plt.savefig('DL_Segmenter_preds.pdf', format='pdf')
print("Total Prediction time: ", time() - start,"secs.")

#######################################################################
"""
img_pred = model.predict(frame.reshape(1,192,256,3)).reshape(192,256)
#plt.imshow(img_pred, plt.cm.binary)

img = cv2.merge((img_pred,img_pred,img_pred))
img = img.reshape(1,192,256,3)

output = ((0.6 * frame) + (0.4 * img)).astype("uint8").reshape(192,256,3)
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.imshow(frame)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(img.reshape(192,256,3))
plt.title('Predicted')
"""