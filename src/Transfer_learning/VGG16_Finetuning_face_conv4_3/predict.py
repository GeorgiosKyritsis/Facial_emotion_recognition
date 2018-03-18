# load necessary modules
import numpy as np
import os
from PIL import Image
from keras.models import load_model
from utils.create_dataset import *

test_directory = '../../../data/ferplus_rgb/Test224/'

# Create numpy arrays X_test, Y_test
X_test, Y_test = create_dataset(test_directory)

X_test[..., 0] -= 93.5940
X_test[...,1] -= 104.7624
X_test[...,2] -= 129.1863

print('X_test shape:', X_test.shape)

print('Loading model...')
model = load_model('../../../logs/VGG16_Finetuning_face_conv4_3/model.hdf5')
print('Model loaded')

print('Get predictions')
y_pred = model.predict(X_test, verbose=1)
