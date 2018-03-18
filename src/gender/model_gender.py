# load necessary modules
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def model_generate(number_of_classes):

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
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-8g'))

    # 9th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-9g'))

    # 10th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-10g'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-4g'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-4g'))

    # 11th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-11g'))

    # 12th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-12g'))

    # 13th Convolutional Layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-13g'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-5g'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-5g'))

    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(units=1024, name='FC-1g'))
    model.add(Activation(activation='relu'))

    # Dropout
    model.add(Dropout(rate=0.5, name='Drop-6g'))

    # 2nd Fully Connected Layer
    model.add(Dense(units=1024, activation='relu', name='FC-2g'))

    # Dropout
    model.add(Dropout(rate=0.5, name='Drop-7g'))

    # Final Layer
    model.add(Dense(units=number_of_classes, activation='softmax', name='output-g'))

    return model
