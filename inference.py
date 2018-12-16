# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:38:06 2018

@author: Kishan Kumar
"""
from main import vgg_model
import cv2
from PIL import Image
import png
import pydicom
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.pyplot import figure
import numpy as np

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras.backend as K


# global variable
top_left_list = []
bottom_right_list = []
prediction = []


def plot_the_result():
    global top_left_list
    global bottom_right_list
    global prediction

    img = cv2.imread('test.png')
    i = 0
    for top_left, bottom_right in zip(top_left_list, bottom_right_list):

        img = cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
        confidence = str(float(prediction[i]) * 100)[:4] + "%"
        img = cv2.putText(img, confidence, bottom_right, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        i = i + 1
    cv2.imwrite('predicted.png',img)
    cv2.imshow('predicted', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def toggle_selector(event):
    toggle_selector.RS.set_active(True)



def line_select_callback(clk, rls):
    global top_left_list
    global bottom_right_list

    top_left_x = int(clk.xdata)
    top_left_y = int(clk.ydata)
    bottom_right_x = int(rls.xdata)
    bottom_right_y = int(rls.ydata)

    top_left_list.append((top_left_x, top_left_y))
    bottom_right_list.append((bottom_right_x, bottom_right_y))



def on_key_press(event):
    global prediction
    if event.key == 'q':
        # get those bounding boxes and crop the images and feed it to the neural network
        # we will take use of PIL
        img = Image.open('test.png')
        # we'll create a dataset of cropped images
        X = np.zeros((len(top_left_list), 50, 50, 1), dtype=np.float32)
        m = 0
        for top_left, bottom_right in zip(top_left_list, bottom_right_list):
            roi = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            cropped_img = img.crop(roi)
            x_img = img_to_array(cropped_img)
            # resize this cropped image to 50X50
            x_img = resize(x_img, (50, 50, 1), mode='constant', preserve_range=True)
            X[m,..., 0] = x_img.squeeze() / 255
            m = m + 1

        # now our dataset is ready to be fed into the model
        input_img = Input((50, 50, 1), name='img')
        model = vgg_model(input_img, dropout=0.5)

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        # load the model
        model.load_weights('model_lung_tumor_detection.h5')
        prediction = model.predict(X, verbose=1)
        np.savetxt('prediction.txt',prediction)
        # Now it's time to show the prediction
        plot_the_result()


# we will ask the user to give the path of the file

if __name__ == '__main__':

    fig, ax = plt.subplots(1)

    # name your testing image as test.png, note it should be a png file and not dicom

    image = cv2.imread('test.png')
    fig.set_dpi(200)
    ax.imshow(image)

    toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
            )
    bbox = plt.connect('key_press_event', toggle_selector)
    key = plt.connect('key_press_event', on_key_press)
    plt.show()
