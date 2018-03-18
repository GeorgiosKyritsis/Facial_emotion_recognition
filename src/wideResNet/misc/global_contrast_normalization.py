import numpy
import scipy
import scipy.misc
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def global_contrast_normalization(in_folder, out_folder, im, s, lmda, epsilon):

	filename = in_folder + im
	X = numpy.array(Image.open(filename))

    # replacement for the loop
	X_average = numpy.mean(X)
	X = X - X_average

    # `su` is here the mean, instead of the sum
	contrast = numpy.sqrt(lmda + numpy.mean(X**2))
	X = s * X / max(contrast, epsilon)

    # scipy can handle it
	scipy.misc.imsave(out_folder + im , X)


train_folder = '../data/Train48/'
valid_folder = '../data/Valid48/'
test_folder = '../data/Test48/'

train_files = os.listdir(train_folder)
valid_files = os.listdir(valid_folder)
test_files = os.listdir(test_folder)

train_images = [file for file in train_files if 'img' in file]
valid_images = [file for file in valid_files if 'img' in file]
test_images = [file for file in test_files if 'img' in file]

# Training set
for img in tqdm(train_images):
	global_contrast_normalization(train_folder, '../data/Train48_contrast/', img, 1, 10, 0.000000001)

# Validation set
for img in tqdm(valid_images):
	global_contrast_normalization(valid_folder, '../data/Valid48_contrast/', img, 1, 10, 0.000000001)

# Test set
for img in tqdm(test_images):
	global_contrast_normalization(test_folder, '../data/Test48_contrast/', img, 1, 10, 0.000000001)




