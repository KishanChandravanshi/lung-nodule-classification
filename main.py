import glob
import cv2
import imutils
import pandas as pd
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

train_folder = 'train/'
# the corresponding file that contains whether the particular image is having tumour or not
train_txt = 'traindatalabels.txt'
validation_txt = 'valdatalabels.txt'
test_txt = 'testdatalabels.txt'
augmented_folder = 'augmented_images/'

# the positive class is sparse, we need to augment the data for positive classes
def augment_image(path_to_image, angle):
    """
    Input: takes path to the image
    Output: Returns the augmented image as a 2d Array
    """
    img = cv2.imread(path_to_image)
    x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # so for each positive image we are actually augmenting three more images so, in total we'll be having 4
    rotated_image = imutils.rotate(x_img, angle)
    return rotated_image


# Why don't create a dataset in the runtime only
def prepare_data(path_to_txt, mode='train'):
    """ it will augment the data for positive class
    as well as create the data matrix
    """
    # so what should be the shape of input vector (m, 50, 50, 1)
    # where m is the total number of training images and 50X50X1 is each training image dimension
    # let's initialise our X matrix
    print('please wait...')
    database = pd.read_csv(path_to_txt)
    # first we need to know how many positive cases are there because
    # then only we can know how many augmented images will be created and the final size of X will be determined

    ############################################ For Training data only
    if mode == 'train':
        n_p = 0
        for data in database.values:
            if data[0].split(' ')[-1] == '1':
                n_p += 1
        print('total postive cases ', str(n_p))
        # so there are n_p number of positive cases
        # thus total size is (database.values + 3 * n_p)

        X = np.zeros((len(database.values) + 3*n_p, 50, 50, 1), dtype = np.float32)
        y = np.zeros((len(database.values) + 3*n_p, 1))
    ##########################################################################
    else:
        X = np.zeros((len(database.values), 50, 50, 1), dtype = np.float32)
        y = np.zeros((len(database.values), 1))

    m = 0
    for data in database.values:
        #print(m)
        filepath, label = data[0].split(' ')

        img = cv2.imread(filepath)

        x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_img = img_to_array(x_img)
        x_img = resize(x_img, (50, 50, 1), mode='constant', preserve_range=True)

        if label == '1':
            # positive class, we need to augment the data
            #print('working with positive class')
            X[m,...,0] = x_img.squeeze() / 255
            y[m,:] = 1
            if mode == 'train':
                for i in range(3):
                    m = m + 1
                    x_img = augment_image(filepath, 90 * (i + 1))
                    x_img = img_to_array(x_img)
                    x_img = resize(x_img, (50, 50, 1), mode='constant', preserve_range=True)
                    X[m, ..., 0] = x_img.squeeze() / 255
                    y[m, :] = 1

            m += 1 # update the index for next image
        else:
            #print('working with negative class')
            X[m,..., 0] = x_img.squeeze() / 255
            y[m, :] = 0
            m += 1

    return X, y



def vgg_model(input_img, dropout = 0.5):

    # block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Dropout(dropout)(x)

    # block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Dropout(dropout)(x)

    # block3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Dropout(dropout)(x)

    # block4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(1, activation='sigmoid', name='prediction')(x)

    model = Model(inputs=[input_img], outputs=[x])
    return model

def custom_model(input_img, dropout = 0.5):

    # block 1
    x = Conv2D(32, (5, 5), activation='relu', padding='same', name='block1_conv1')(input_img)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Dropout(dropout)(x)
    # block 2
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Dropout(dropout)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', name='prediction')(x)

    model = Model(inputs=[input_img], outputs=[x])
    return model

if __name__ == '__main__':
    X_train, y_train = prepare_data(train_txt, 'train')

    X_val, y_val = prepare_data(validation_txt, 'val')


    input_img = Input((50, 50, 1), name='img')
    model = vgg_model(input_img, dropout=0.5)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()


    callbacks = [
            EarlyStopping(patience=20, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint('model_lung_tumor_detection.h5', verbose=1, save_best_only=True,
                            save_weights_only=True)
            ]

    results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                        validation_data=(X_val, y_val))

    # Load best model
    X_test, y_test = prepare_data(test_txt , 'test')
    model.load_weights('model_lung_tumor_detection.h5')
    # Evaluate on validation set (this must be equals to the best log_loss)
    model.evaluate(X_test, y_test, verbose=1)
