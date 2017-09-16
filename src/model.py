from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
from keras.layers import Cropping2D

def model(input_size):
    """
    Model to do behavioral cloning on data generated with udacity's simulator
    Initial model is taken from the Nvidia paper (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
    :param input_size: input size (height, width, channels)
    :return: keras model object
    """
    model = Sequential()
    # Cropping image to remove top and bottom pixels of the images
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=input_size))
    # Normalize image data
    model.add(Lambda(lambda x:(x / 255) - 0.5))

    # Convolution layers
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    #Fully connected layers
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model