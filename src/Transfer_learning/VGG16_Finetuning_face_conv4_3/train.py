# load necessary modules
import numpy as np
import os
from vgg import VGG
from keras_vggface.vggface import VGGFace
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.optimizers import SGD
from callbacks import *
from utils.create_dataset import *
from keras import optimizers
from numpy import random

# Create a generator for the training set because
# otherwise we run oout of GPU memory
def generator(batch_size, directory):
 # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((batch_size, 224, 224, 3))
     batch_labels = np.zeros((batch_size,8))
     while True:
        for i in range(batch_size):
          # choose random index in features
          index= random.choice(27947, 1)[0]
          img_path = directory + 'img' + str(index) + '.png'
          lab_path = directory + 'lab' + str(index) + '.npy'
          im = image.load_img(img_path, target_size=(224, 224))
          x = image.img_to_array(im)
          x = np.expand_dims(x, axis=0)
          #x = preprocess_input(x)

          y = np.load(directory + 'lab' + str(i) + '.npy')

          batch_features[i] = x
          batch_labels[i] = y

        batch_features[..., 0] -= 93.5940
        batch_features[...,1] -= 104.7624
        batch_features[...,2] -= 129.1863

        yield batch_features, batch_labels

# Create a generator for the validation set because
# otherwise we run oout of GPU memory
def generator_val(batch_size, directory):
 # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((batch_size, 224, 224, 3))
     batch_labels = np.zeros((batch_size,8))
     index = 0
     while True:
        for i in range(batch_size):
          if index == 3508:
               index = 0
          # choose random index in features
          img_path = directory + 'img' + str(index) + '.png'
          lab_path = directory + 'lab' + str(index) + '.npy'
          im = image.load_img(img_path, target_size=(224, 224))
          x = image.img_to_array(im)
          x = np.expand_dims(x, axis=0)
          y = np.load(directory + 'lab' + str(i) + '.npy')

          batch_features[i] = x
          batch_labels[i] = y
          index+=1
        
        batch_features[..., 0] -= 93.5940
        batch_features[...,1] -= 104.7624
        batch_features[...,2] -= 129.1863
        yield batch_features, batch_labels


# Read files
train_directory = '../../../data/ferplus_rgb/Train224/'
valid_directory = '../../../data/ferplus_rgb/Valid224/'

# input image dimensions
img_rows, img_cols = 224, 224
# grayscale images
img_channels = 3
nb_classes = 8

verbose = 1

# hyperparameters
nb_epoch = 51
batch_size = 128

# set up th ecallbacks
c1 = CheckpointCallback1()
c2 = MetricsCallback()
callbacks = [c1, c2]
vgg_model = VGGFace(model='vgg16', include_top=False, pooling='max', input_shape=(224, 224, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 14 layers.
for layer in vgg_model.layers[:14]:
    layer.trainable = False

#Adding custom Layers
last_layer = vgg_model.get_layer('pool5').output
x = Flatten()(last_layer)
x = Dense(4096, activation="relu")(x)
x = Dense(4096, activation="relu")(x)
predictions = Dense(nb_classes, activation="softmax")(x)

model_final = Model(input = vgg_model.input, output = predictions)

sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model_final.compile(loss='categorical_crossentropy', optimizer=sgd)

print(model_final.summary())


model_final.fit_generator(generator(batch_size, train_directory),
                         steps_per_epoch=27947//batch_size,
                         nb_epoch=nb_epoch,
                         verbose=1,
                         callbacks=callbacks,
                         validation_data=generator_val(batch_size, valid_directory),
                         validation_steps=3508//batch_size)
