import os
import numpy as np
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image


def create_dataset(directory):
    files = os.listdir(directory)
    images = [file for file in files if 'img' in file]
    #labels = [file for file in files if 'lab' in file]

    features = np.zeros((len(images), 224, 224, 3))
    labels = np.zeros((len(images),8))

    for i in range(len(features)):
        img_path = directory + 'img' + str(i) + '.png'
        lab_path = directory + 'lab' + str(i) + '.npy'
        im = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        y = np.load(directory + 'lab' + str(i) + '.npy')

        features[i] = x
        labels[i] = y

    return features, labels

'''
    X = []
    for img in images:
		#img_path = input_folders[i] + img
        img_path = directory + img
        im = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X.append(x)
	
    X = np.concatenate([arr[np.newaxis] for arr in X])
    X = X.astype('float32')

    Y = [np.load(directory + 'lab' + str(i) + '.npy') for i in range(len(labels))]
    Y = np.concatenate([arr[np.newaxis] for arr in Y])
	
    return X, Y
'''
