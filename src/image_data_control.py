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
    path : str
        [File location of the images directory. i.e : ".../images/"]
    
    Returns
    -------
    images_dataset_list : numpy array
        [All the images from the directory loaded into a variable]
    class_num : numpy array
        [An array of each class directory name. Numbered 0-42. ]
    shapes : list
        [A list of the shape of each image.]
    class_count : int
        [The count of the current directory where the image is located.]
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
    This function saves variables as binary pickle files for reuse later.
    
    Parameters
    ----------
    var_ : [Type subject to change according to what variable is chosen.]
        [The variable you whose contents you want to save.]
    file_name : str
        [Name of the pickle file. i.e : 'class_num.pickle']

    Returns
    -------
    NONE
        [Generates a file in the local directory.]
    '''
    
    with open(file_name, 'wb') as f:
        pickle.dump(var_, f)
        
def load_variables(file_name : str):
    
    '''
    This function loads variables from a binary pickle file.
    
    Parameters
    ----------
    file_name : str
        [File name. i.e : 'images_dataset_list.pickle']
    
    Returns
    -------
    var_ - [Type subject to change according to what pickel file is chosen.]
      [A variable now populated with the contents of the pickle file.]
    '''
    
    with open(file_name, 'rb') as f:
        var_ = pickle.load(f)

    return var_


def data_split(images_dataset, class_num):
    
    '''
    This function splits our image data set with a train test split.
    
    Parameters
    ----------
    images_dataset : numpy array
        [A nested array whose elements are the image arrays.]
    class_num : numpy array
        [The number refers to which directory/class the image was found in.]
    
    Returns
    -------
    X_train - pandas Array or pandas Dataframe
      [Training set.]
    X_test - pandas Array or pandas Dataframe
      [Test set.]
    y_train - series
      [Training set's correct labels.]
    y_test - series
      [Test set's correct labels.]
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(images_dataset, class_num, test_size = 0.2)

    return X_train, X_test, y_train, y_test


def train_validation_split(X_train, y_train, validation_size = 0.2):
    
    '''
    This function splits our image training sets with a train validation split.

    Parameters
    ----------
    X_train : pandas Array or pandas Dataframe
        [Feature matrix]
    y_train : pandas Series
        [Target array]
    validation_size : float
        [Size of the validation set. Defaulted to 0.20]
    
    Returns
    -------
    X_train - pandas Array or pandas Dataframe
      [Training set.]
    X_validation - pandas Array or pandas Dataframe
      [Validation set.]
    y_train - series
      [Training set's correct labels.]
    y_validation - series
      [Validation set's correct labels.]
    '''
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validation_size)

    return X_train, X_validation, y_train, y_validation 


def labels_df(label_file):
    
    '''
    This function loads the labels into a dataframe from a csv file.
    
    Parameters
    ----------
    label_file : str
        [File path of the file that contain the labels]
    
    Returns
    -------
    data - pandas dataframe
      [A pandas dataframe populated with the file's contents.]
    '''
    
    data = pd.read_csv(label_file)
    
    return data


