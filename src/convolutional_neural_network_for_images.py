## Tensoflow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer, Dense, \
                                    Conv2DTranspose, MaxPooling2D, Dropout, Flatten, Input
from keras.models import Sequential

    
def cnn_img(X_train, y_train, X_validation, y_validation, early_stopping, model_checkpoint):
    '''
    This function takes the image data and runs a convolutional neural network.
    
    Parameters
    ----------
    X_train : pandas Array or pandas Dataframe
        [Feature matrix]
    y_train : pandas Series
        [Target array]
    X_validation : pandas Array or pandas Dataframe
        [Feature matrix]
    y_validation : pandas Series
        [Target array]
    early_stopping : keras.callbacks.EarlyStopping
        [Early Stopping which should be defined before using fucntion.]
    model_checkpoint : keras.callbacks.ModelCheckpoint
        [Model Checkpoints which should define before using function.]
    
    Returns
    -------
    model : h5 file
        [Saves the model in a file.]
    history : h5 file
        [Saves the historical model in a file, factoring, early-stopping and model checkpoints.]

    '''
    # Model 

    model = keras.models.Sequential()
    model.add((Conv2D(60, (5,5), input_shape = (imageDim[0], imageDim[1], 1), activation = 'relu'))) 
    model.add((Conv2D(60, (5,5), activation = 'relu')))
    model.add(MaxPooling2D(pool_size = (2, 2))) 

    model.add((Conv2D(60 // 2, (3,3), activation = 'relu')))
    model.add((Conv2D(60 // 2, (3,3), activation = 'relu')))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(class_count, activation = 'softmax')) # ouput layer

    # COMPILE MODEL
    model.compile(Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Fit
    batch_size_val = 50
    steps_per_epoch_val = len(X_train) // batch_size_val
    epochs_val = 50

    history = model.fit(X_train, y_train, 
                        batch_size = batch_size_val,
                        steps_per_epoch = steps_per_epoch_val,
                        epochs = epochs_val,
                        shuffle = 1, 
                        validation_data = (X_validation,y_validation),
                        callbacks = [early_stopping, model_checkpoint], verbose = 1)
    
    print(model.summary())
    
    # Save
    model.save('drive/MyDrive/Capstone3/TrafficSigns/trafficSignModel.h5')
    model.save_weights('drive/MyDrive/Capstone3/TrafficSigns/trafficSignModelweights.h5')
    
    return model, history








