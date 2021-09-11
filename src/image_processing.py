## Standard
import numpy as np
import pandas as pd
import cv2

## Image Related
from PIL import Image

## Tensoflow and Keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Functions

def grayscale(img):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a Random Forest model using them.
    
    Parameters
    ----------
    img : pandas Array or pandas Dataframe
        [Feature matrix]
        
    Returns
    -------
    img_expanded : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000.]
    '''
    img = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_expanded = gray[:, :, np.newaxis]
    return img_expanded

def equalize(img):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a Random Forest model using them.
    
    Parameters
    ----------
    img : pandas Array or pandas Dataframe
        [Feature matrix]

    Returns
    -------
    img : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000.]
    '''
    img1 = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    img = cv2.equalizeHist(img1)
    return img

def preprocessing(img):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a Random Forest model using them.
    
    Parameters
    ----------
    img : pandas Array or pandas Dataframe
        [Feature matrix]

    Returns
    -------
    img : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000.]
    '''
    img = grayscale(img)     
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255    
    return img

def depth_adder(X_train, X_validation, X_test):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a Random Forest model using them.
    
    Parameters
    ----------
    X_train : pandas Array or pandas Dataframe
        [Feature matrix]
    X_validation : pandas Series
        [Target array]
    X_test : kwargs for `RandomForestClassifier()`
    
    Returns
    -------
    X_train : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000.]
    X_validation : numpy array
        [X_test made from train_test_split]
    X_test : pandas Series
        [y_test made from train_test_split]
    '''
    X_train = X_train.reshape(X_train.shape[0], 
                            X_train.shape[1], 
                            X_train.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], 
                                      X_validation.shape[1], 
                                      X_validation.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0],
                          X_test.shape[1],
                          X_test.shape[2], 1)

    return X_train, X_validation, X_test

