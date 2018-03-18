import os
import numpy as np
from PIL import Image

# Create a dataset from transfer learning features
# to give as input to SVM
def create_dataset(directory, directory_labels):
    files = os.listdir(directory)
    images = [file for file in files if 'img' in file]
    files_lab = os.listdir(directory_labels)
    labels = [file for file in files_lab if 'lab' in file]

    X = []
    print('creating X')
    for i in range(len(images)):
    	tmp = np.load(directory + 'img' + str(i) + '.npy')
    	le = tmp.shape[1]
    	tmp = tmp[0,:]
    	X.append(tmp.tolist())

    X = np.array(X)
    
    labs = []
    print('creating Y')
    for i in range(len(labels)):
    	tmp = np.load(directory_labels + 'lab' + str(i) + '.npy')
    	m = np.max(tmp)
    	indexes = [i for i, j in enumerate(tmp) if j == m]
    	labs.append(indexes[-1])

    Y = np.array(labs)
    
    print(X.shape, Y.shape)
    return X, Y
