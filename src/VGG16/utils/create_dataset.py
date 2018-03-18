import os
import numpy as np
from PIL import Image

def create_dataset(directory):
    files = os.listdir(directory)
    images = [file for file in files if 'img' in file]
    labels = [file for file in files if 'img' in file]

    X = [np.reshape(np.array(Image.open(directory + 'img' + str(i) + '.png')), (48,48,1)) for i in range(len(images))]
    X = np.concatenate([arr[np.newaxis] for arr in X])
    X = X.astype('float32')

    Y = [np.load(directory + 'lab' + str(i) + '.npy') for i in range(len(labels))]
    Y = np.concatenate([arr[np.newaxis] for arr in Y])

    return X, Y
