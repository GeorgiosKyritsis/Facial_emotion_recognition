import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def model_generate_emotion(number_of_classes):

    image_rows = 64
    image_cols = 64

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='Conv-1',
                     input_shape=(image_rows, image_cols, 1)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='Conv-2'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-1'))

    # Dropout Layer
    model.add(Dropout(rate=0.25, name='Drop-1'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv-3'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv-4'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-2'))

    # Dropout Layer
    model.add(Dropout(rate=0.25, name='Drop-2'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='Conv-5'))

    # 6th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='Conv-6'))

    # 7th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-7'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-3'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-3'))

    # 8th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-8'))

    # 9th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-9'))

    # 10th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-10'))

    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool-4'))

    # Dropout
    model.add(Dropout(rate=0.25, name='Drop-4'))

    model.add(Flatten())

    # 1st Fully Connected Layer
    model.add(Dense(units=1024, activation='relu', name='FC-1'))

    # Dropout
    model.add(Dropout(rate=0.5, name='Drop-5'))

    # 2nd Fully Connected Layer
    model.add(Dense(units=1024, activation='relu', name='FC-2'))

    # Dropout
    model.add(Dropout(rate=0.5, name='Drop-6'))

    # Final Layer
    model.add(Dense(units=number_of_classes, activation='softmax', name='output'))

    return model
