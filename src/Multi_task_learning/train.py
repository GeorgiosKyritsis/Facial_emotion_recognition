import numpy as np
import os
import random
import math
from sklearn.metrics import mean_absolute_error
from keras.preprocessing.image import ImageDataGenerator
from model import model_generate
from keras import optimizers


# Helper functions
def get_emotion_accuracy(model, X, Y):

    Y_pred = model.predict(X)
    Y_pred_arg = np.argmax(Y_pred, axis=1)

    count = 0
    for i in range(Y.shape[0]):
        if (Y[i][Y_pred_arg[i]] == np.max(Y[i])):
            count += 1

    accuracy = count / Y.shape[0]

    return accuracy

def get_gender_accuracy(model, X, Y):

    Y_pred = model.predict(X)
    Y_pred_arg = np.argmax(Y_pred, axis=1)

    Y_arg = np.argmax(Y, axis=1)

    count = np.sum(Y_arg == Y_pred_arg)
    accuracy = count / Y.shape[0]

    return accuracy

def get_age_mae(model, X, Y):

    Y_pred = model.predict(X)
    mae = mean_absolute_error(Y, Y_pred)

    return mae


## Read Emotion data
X_train_emotion = np.load('../../data/multitask_learning/Emotion/Train/final_data_aug.npy')
Y_train_emotion = np.load('../../data/multitask_learning/Emotion/Train/final_labels_data_aug.npy')

X_valid_emotion = np.load('../../data/multitask_learning/Emotion/Valid/final_data.npy')
Y_valid_emotion = np.load('../../data/multitask_learning/Emotion/Valid/final_labels_data.npy')


## Read Gender data
X_train_gender = np.load('../../data/multitask_learning/Gender/Train/final_data.npy')
Y_train_gender = np.load('../../data/multitask_learning/Gender/Train/final_labels_data.npy')

X_valid_gender = np.load('../../data/multitask_learning/Gender/Valid/final_data.npy')
Y_valid_gender = np.load('../../data/multitask_learning/Gender/Valid/final_labels_data.npy')


## Read Age data
X_train_age = np.load('../../data/multitask_learning/Age/Train/final_data.npy')
Y_train_age = np.load('../../data/multitask_learning/Age/Train/final_labels_data.npy')

X_valid_age = np.load('../../data/multitask_learning/Age/Valid/final_data.npy')
Y_valid_age = np.load('../../data/multitask_learning/Age/Valid/final_labels_data.npy')


if __name__ == '__main__':

    # parameters
    batch = 256
    epochs = 10000
    verbose = 1
    input_shape = (64, 64, 1)
    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.9

    # Generate the architecture
    emotion_model, gender_model, age_model = model_generate()

    # Stochastic gradient descent optimizer
    sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    # Configure the learning process by compiling the network
    emotion_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    gender_model.compile(optimizer='adam', loss='categorical_crossentropy')
    age_model.compile(optimizer='adam', loss='mean_absolute_error')

    emotion_accuracy = - math.inf
    gender_accuracy = - math.inf
    age_mae = math.inf

    # Train the model for a fixed number of epochs
    for epoch in range(epochs):
        print('Epoch:', epoch)
        indices_emotion = list(range(X_train_emotion.shape[0]))
        indices_age = list(range(X_train_age.shape[0]))
        indices_gender = list(range(X_train_gender.shape[0]))

        for iteration in range(int(X_train_emotion.shape[0] / batch)):
            print('Iteration:', iteration)
            if (len(indices_emotion) >= batch):
                ind_emotion = random.sample(indices_emotion, batch)
                indices_emotion = list(set(indices_emotion) - set(ind_emotion))
            else:
                ind_emotion = [random.randint(0,X_train_emotion.shape[0]-1) for i in range(batch)]

            if (len(indices_age) >= batch):
                ind_age = random.sample(indices_age, batch)
                ind_gender = random.sample(indices_gender, batch)
                indices_age = list(set(indices_age) - set(ind_age))
                indices_gender = list(set(indices_gender) - set(ind_gender))
            else:
                ind_age = [random.randint(0,X_train_age.shape[0]-1) for i in range(batch)]
                ind_gender = [random.randint(0,X_train_gender.shape[0]-1) for i in range(batch)]

            X_batch_emotion = X_train_emotion[ind_emotion]
            Y_batch_emotion = Y_train_emotion[ind_emotion]

            X_batch_gender = X_train_gender[ind_gender]
            Y_batch_gender = Y_train_gender[ind_gender]

            X_batch_age = X_train_age[ind_age]
            Y_batch_age = Y_train_age[ind_age]

            emotion_model.train_on_batch(X_batch_emotion, Y_batch_emotion)
            gender_model.train_on_batch(X_batch_gender, Y_batch_gender)
            age_model.train_on_batch(X_batch_age, Y_batch_age)

        # Calculate the accuracy for Emotion, gender, age validation datasets
        emotion_acc = get_emotion_accuracy(emotion_model, X_valid_emotion, Y_valid_emotion)
        gender_acc = get_gender_accuracy(gender_model, X_valid_gender, Y_valid_gender)
        age_m = get_age_mae(age_model, X_valid_age, Y_valid_age)

        # Save weights if the metric improves
        if emotion_acc > emotion_accuracy:
            emotion_accuracy = emotion_acc
            file = '../../logs/multitask_learning/' + 'weights-emotion.hdf5'
            emotion_model.save_weights(file)

        if gender_acc > gender_accuracy:
            gender_accuracy = gender_acc
            file = '../../logs/multitask_learning/' + 'weights-gender.hdf5'
            gender_model.save_weights(file)

        if age_m < age_mae:
            age_mae = age_m
            file = '../../logs/multitask_learning' + 'weights-age.hdf5'
            age_model.save_weights(file)
