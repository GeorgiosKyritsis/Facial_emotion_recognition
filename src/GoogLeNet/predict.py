import numpy as np
import os
from keras.models import load_model
from utils.create_dataset import *

model_name = 'GoogLeNet'

test_directory = '../../data/ferplus/Test48/'

X_test, Y_test = create_dataset(test_directory)
X_test = X_test/255.0

print('X_test shape:', X_test.shape)

print('Loading model...')
model = load_model('../../logs/{}/model.hdf5'.format(model_name))
print('Model loaded')

print('Get predictions')
y_pred = model.predict(X_test, verbose=1)
