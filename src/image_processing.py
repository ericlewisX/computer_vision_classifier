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
    Takes in an image and returns its grayscaled transformation.
    
    Parameters
    ----------
    img : array
        [The image file.]
        
    Returns
    -------
    img_expanded : array
        [The grayscaled image.]
    '''
    img = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_expanded = gray[:, :, np.newaxis]
    return img_expanded

def equalize(img):
    '''
    This function takes in an image and redistributes the brightness values of the pixels in an image so the image more evenly represents the entire range of brightness levels.
    
    Parameters
    ----------
    img : array
        [The image file.]
        
    Returns
    -------
    img : array
        [The equalized image.]
    '''
    img1 = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
    img = cv2.equalizeHist(img1)
    return img

def preprocessing(img):
    '''
    This function takes in an image and returns an image that is greyscaled, equalized and standardized.
    
    Parameters
    ----------
    img : array
        [The image file.]
        
    Returns
    -------
    img : array
        [The fully preprocessed image.]
    '''
    img = grayscale(img)     
    img = equalize(img)     
    img = img / 255    
    return img

def depth_adder(X_train, X_validation, X_test):
    '''
    This function takes in the X_train, X_validation, & X_test sets and adds another dimension to each image.
    
    Parameters
    ----------
    X_train : array
        [Feature matrix]
    X_validation : arra
        [Target array]
    X_test : array
    
    Returns
    -------
    X_train : image array
        [An array of all the images in the training set now with one extra dimension.]
    X_validation : image array
        [An array of all the images in the validation set now with one extra dimension.]
    X_test : pandas Series
        [An array of all the images in the test set now with one extra dimension.]
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

