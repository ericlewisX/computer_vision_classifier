# Imports #
import pandas as pd
import numpy as np

# Machine Learning
import classifier_base as CLFScores

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.tree import RandomForestClassifier
from sklearn.inspection import permutation_importance


# Functions

def load_data(path):
    '''
    This function loads all the images from all the directories into lists.
    
    Parameters
    ----------
    X : pandas Array or pandas Dataframe
        [Feature matrix]
    y : pandas Series
        [Target array]
    **kwargs : kwargs for `RandomForestClassifier()`
    
    Returns
    -------
    clf : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000.]
    X_test : numpy array
        [X_test made from train_test_split]
    y_test : pandas Series
        [y_test made from train_test_split]
    '''
    class_directory_num, image_list, class_, shapes = 0, [], [], []
    list_ = os.listdir(path)     # Number of directories in path is synonymous with number of classes
    class_count = len(list_)

    for folder in range(len(list_)):

        sign_list = os.listdir(path + '/' + str(class_directory_num))

        for x in sign_list:

            image = cv2.imread(path + '/' + str(class_directory_num) + '/' + x)
            image_list.append(image)
            class_.append(class_directory_num)
            shapes.append(image.shape)

        print(class_directory_num, end = ' ')
        class_directory_num += 1

    images_dataset_list = np.array(image_list)
    class_num = np.array(class_)

    return images_dataset_list, class_num, shapes, class_count

def save_variables(var_, file_name : str):
    '''
    [This function splits our data with a train validation split. ]

    Returns
    -------
    [type] - X_train
      [description]
    [type] - X_validation
      [description]
    [type] - y_train
      [description]
    [type] - y_validation
      [description]
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(var_, f)
        
def load_variables(file_name : str):
    '''
    [This function splits our data with a train validation split. ]

    Returns
    -------
    [type] - X_train
      [description]
    [type] - X_validation
      [description]
    [type] - y_train
      [description]
    [type] - y_validation
      [description]
    '''
    with open(file_name, 'rb') as f:
        var_ = pickle.load(f)

    return var_


def data_split(images_dataset, class_num):
    '''
    [This function splits our data with a train validation split. ]

    Returns
    -------
    [type] - X_train
      [description]
    [type] - X_validation
      [description]
    [type] - y_train
      [description]
    [type] - y_validation
      [description]
    '''
    X_train, X_test, y_train, y_test = train_test_split(images_dataset, class_num, test_size=0.2)

    return X_train, X_test, y_train, y_test


def train_validation_split(X_train, y_train, validation_size = 0.2):
    '''
    [This function splits our data with a train validation split. ]

    Returns
    -------
    [type] - X_train
      [description]
    [type] - X_validation
      [description]
    [type] - y_train
      [description]
    [type] - y_validation
      [description]
    '''
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_size)

    return X_train, X_validation, y_train, y_validation 

def labels_df(label_file):
    '''
    [This function splits our data with a train validation split. ]

    Returns
    -------
    [type] - X_train
      [description]
    [type] - X_validation
      [description]
    [type] - y_train
      [description]
    [type] - y_validation
      [description]
    '''
    data = pd.read_csv(label_file)
    return data


