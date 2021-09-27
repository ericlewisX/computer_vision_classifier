## Standard
import numpy as np
import pandas as pd
import os
import cv2
import pickle

## Custom Functions
from image_data_control import load_data, save_variables, labels_df


path = "imagedataset/TrafficSignClassifier/data/myData/images/" # Directory for Images
label_csv = 'imagedataset/TrafficSignClassifier/labels.csv'     # Labels for Directories

if __name__ == "__main__":
    
    # Load Data from path directory
    images_dataset_list, class_num, shapes, class_count = load_data(path)
    
    # Save Variables
    save_variables(images_dataset_list, 'images_dataset_list.pickle')
    save_variables(class_num, 'class_num.pickle')
    save_variables(shapes, 'shapes.pickle')
    save_variables(class_count, 'class_count.pickle')
    
    # Load Labels
    data = labels_df(label_csv)
    save_variables(data, 'data_label_csv.pickle')