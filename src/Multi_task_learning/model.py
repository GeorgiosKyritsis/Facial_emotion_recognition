import numpy as np
from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def model_generate():

    image_rows = 64
    image_cols = 64

    ####### Shared model #######
    ############################
    shared_model = Sequential()

    # 1st Convolutional Layer
    shared_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='Conv-1',
                     input_shape=(image_rows, image_cols, 1)))

    # 2nd Convolutional Layer
    shared_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='Conv-2'))

    # Max Pooling
    shared_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-1'))

    # Dropout Layer
    shared_model.add(Dropout(rate=0.25, name='Drop-1'))

    # 3rd Convolutional Layer
    shared_model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv-3'))

    # 4th Convolutional Layer
    shared_model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='Conv-4'))

    # Max Pooling
    shared_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-2'))

    # Dropout Layer
    shared_model.add(Dropout(rate=0.25, name='Drop-2'))

    # 5th Convolutional Layer
    shared_model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='Conv-5'))

    # 6th Convolutional Layer
    shared_model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='Conv-6'))

    # 7th Convolutional Layer
    shared_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-7'))

    # Max Pooling
    shared_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-3'))

    # Dropout
    shared_model.add(Dropout(rate=0.25, name='Drop-3'))

    ####### End of Shared model #######
    ###################################

    ####### 1st Branch #######
    ######## Emotion ########
    emotion_model = Sequential()

    emotion_model.add(shared_model)
    # 8th Convolutional Layer
    emotion_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-8'))

    # 9th Convolutional Layer
    emotion_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-9'))

    # 10th Convolutional Layer
    emotion_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-10'))

    # Max Pooling
    emotion_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Pool-4'))

    # Dropout
    emotion_model.add(Dropout(rate=0.25, name='Drop-4'))

    emotion_model.add(Flatten())
    # 1st Fully Connected Layer
    emotion_model.add(Dense(units=1024, name='FC-1'))
    emotion_model.add(Activation(activation='relu'))

    # Dropout
    emotion_model.add(Dropout(rate=0.5, name='Drop-5'))

    # 2nd Fully Connected Layer
    emotion_model.add(Dense(units=1024, activation='relu', name='FC-2'))

    # Dropout
    emotion_model.add(Dropout(rate=0.5, name='Drop-6'))

    # Final Layer
    emotion_model.add(Dense(units=8, activation='softmax', name='output'))

    ####### 2nd Branch #######
    ######### Gender #########
    gender_model = Sequential()

    gender_model.add(shared_model)

    # 8th Convolutional Layer
    gender_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-8g'))

    # 9th Convolutional Layer
    gender_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-9g'))

    # 10th Convolutional Layer
    gender_model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-10g'))

    # Max Pooling
    gender_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-4g'))

    # Dropout
    gender_model.add(Dropout(rate=0.25, name='Drop-4g'))

    # 11th Convolutional Layer
    gender_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-11g'))

    # 12th Convolutional Layer
    gender_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-12g'))

    # 13th Convolutional Layer
    gender_model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-13g'))

    # Max Pooling
    gender_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-5g'))

    # Dropout
    gender_model.add(Dropout(rate=0.25, name='Drop-5g'))

    gender_model.add(Flatten())
    # 1st Fully Connected Layer
    gender_model.add(Dense(units=1024, name='FC-1g'))
    gender_model.add(Activation(activation='relu'))

    # Dropout
    gender_model.add(Dropout(rate=0.5, name='Drop-6g'))

    # 2nd Fully Connected Layer
    gender_model.add(Dense(units=1024, activation='relu', name='FC-2g'))

    # Dropout
    gender_model.add(Dropout(rate=0.5, name='Drop-7g'))

    # Final Layer
    gender_model.add(Dense(units=2, activation='softmax', name='output-g'))

    ####### 3nd Branch #######
    ######### Age #########
    age_model = Sequential()

    age_model.add(shared_model)

    # 8th Convolutional Layer
    age_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-8a'))

    # 9th Convolutional Layer
    age_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-9a'))

    # 10th Convolutional Layer
    age_model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-10a'))

    # Max Pooling
    age_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-4a'))

    # Dropout
    age_model.add(Dropout(rate=0.25, name='Drop-4a'))

    # 11th Convolutional Layer
    age_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-11a'))

    # 12th Convolutional Layer
    age_model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='Conv-12a'))

    # 13th Convolutional Layer
    age_model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='Conv-13a'))

    # Max Pooling
    age_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Pool-5a'))

    # Dropout
    age_model.add(Dropout(rate=0.25, name='Drop-5a'))

    age_model.add(Flatten())
    # 1st Fully Connected Layer
    age_model.add(Dense(units=1024, name='FC-1a'))
    age_model.add(Activation(activation='relu'))

    # Dropout
    age_model.add(Dropout(rate=0.5, name='Drop-6a'))

    # 2nd Fully Connected Layer
    age_model.add(Dense(units=1024, activation='relu', name='FC-2a'))

    # Dropout
    age_model.add(Dropout(rate=0.5, name='Drop-7a'))

    # Final Layer
    age_model.add(Dense(units=1, name='output-a'))

    return emotion_model, gender_model, age_model
