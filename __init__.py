## Standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

import sys
sys.setrecursionlimit(1500)

## Image Related
from PIL import Image
from skimage.io import imsave, imshow

## Sklearn
from sklearn.model_selection import train_test_split

## Tensoflow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, Dense, \
                                    Conv2DTranspose, MaxPooling2D, Dropout, Flatten, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

## Custom Functions
from data_control import load_variables, data_split, train_validation_split
from image_processing import grayscale, equalize, preprocessing, depth_adder
from convolutional_neural_network_for_images import cnn_img


path = "imagedataset/TrafficSignClassifier/data/myData/images/"               # Directory for Images
label_csv = 'imagedataset/TrafficSignClassifier/labels.csv'                   # Labels for Directories


if __name__ == "__main__":
       
    # Load Variables
    images_dataset_list = load_variables('images_dataset_list.pickle')
    class_num = load_variables('class_num.pickle')
    shapes = load_variables('shapes.pickle')
    class_count = load_variables('class_count.pickle')
    
    # Train - Test - Validation - Splits
    X_train, X_test, y_train, y_test  = data_split(images_dataset_list, class_num)
    X_train, X_validation, y_train, y_validation = train_validation_split(X_train, y_train)
    
    # Load Labels
    data = load_variables('data_label_csv.pickle')
    
    ## Image Processing
    X_train = np.array(list(map(preprocessing, X_train)))
    X_validation = np.array(list(map(preprocessing, X_validation)))
    X_test = np.array(list(map(preprocessing, X_test)))
    
    
    X_train, X_validation, X_test = depth_adder(X_train, X_validation, X_test)
    
    
    # Image augmenter
    augmenter = ImageDataGenerator(width_shift_range = 0.1, 
                                height_shift_range = 0.1,
                                zoom_range = 0.2, 
                                shear_range = 0.1,
                                rotation_range = 10) 
    augmenter.fit(X_train)
    batches = augmenter.flow(X_train, y_train, batch_size = 20) 
    X_batch, y_batch = next(batches)
    
    
    y_train, y_validation, y_test = to_categorical(y_train, class_count), to_categorical(y_validation, class_count), to_categorical(y_test, class_count)

    
    # CNN MOdel
    cnn_img(X_train, y_train, X_validation, y_validation, early_stopping, model_checkpoint)
    
    
    
    
    
    
    