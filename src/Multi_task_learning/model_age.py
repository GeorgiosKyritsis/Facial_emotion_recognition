import numpy as np
from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def model_generate_age(number_of_classes):

    image_rows = 64
    image_cols = 64

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='Conv-1',
                     input_shape=(image_rows, image_cols, 1), trainable=False))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='Conv-2', trainable=False))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-1'))

    # Dropout Layer
    model.add(Dropout(rate=0.25, name='Drop-1'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv-3', trainable=False))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv-4', trainable=False))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-2'))

    # Dropout Layer
    model.add(Dropout(rate=0.25, name='Drop-2'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='Conv-5', trainable=False))

    # 6th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='Conv-6', trainable=False))

    # 7th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-7', trainable=False))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-3'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-3'))

    # 8th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-8a'))

    # 9th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-9a'))

    # 10th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-10a'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-4a'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-4a'))

    # 11th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-11a'))

    # 12th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-12a'))

    # 13th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-13a'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-5a'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-5a'))

    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(units=1024, name='FC-1a'))
    model.add(Activation(activation='relu'))

    # Dropout
    model.add(Dropout(rate=0.5, name='Drop-6a'))

    # 2nd Fully Connected Layer
    model.add(Dense(units=1024, activation='relu', name='FC-2a'))

    # Dropout
    model.add(Dropout(rate=0.5, name='Drop-7a'))

    # Final Layer
    model.add(Dense(units=number_of_classes, name='output-a'))

    return model
