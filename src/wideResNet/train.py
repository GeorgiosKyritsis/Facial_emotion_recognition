# load necessary modules
import numpy as np
import os
from PIL import Image
import wide_residual_network as wrn
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from callbacks import *
from utils.create_dataset import *


# Read files
train_directory = '../../data/ferplus/Train48/'
valid_directory = '../../data/ferplus/Valid48/'

X_train, Y_train = create_dataset(train_directory)
X_valid, Y_valid = create_dataset(valid_directory)
X_valid = X_valid / 255.0

# Real time augmentation
generator = ImageDataGenerator(rotation_range=10,
								width_shift_range=5./48,
								height_shift_range=5./48,
								horizontal_flip=True,
								rescale=1./255)

generator.fit(X_train, seed=0, augment=True)

# input image dimensions
img_shape = (48,48,1)

verbose = 1

# hyperparameters
nb_epoch = 300
batch_size = 128
lr_schedule = [100, 150]

# Custom scheduler for the learning rate
def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02
    else:
        return 0.004


# Set up the callbacks
metrics_callback = MetricsCallback(validation_data=(X_valid, Y_valid))
checkpoint_callback = CheckpointCallback(monitor='accuracy', verbose=verbose, save_weights_only=False)
lr_callback = LearningRateScheduler(schedule=schedule)

callbacks = [metrics_callback, checkpoint_callback, lr_callback]


#For WRN-28-10 N = (nb_layers-4)/6 = (28-4)/6 = 4, k=10  (maybe memory constraints!!!) 4.17%
#For WRN-16-8 N = (nb_layers-4)/6 = (16-4)/6 = 2, k=8 4.81%
#For WRN-40-4 N = (nb_layers-4)/6 = (16-4)/6 = 2, k=8 4.97%

# Set up the architecture
model = wrn.create_wide_residual_network(img_shape, nb_classes=8, N=4, k=10, dropout=0.3)

# set the optimizer
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Model Summary
print(model.summary())

# start training
model.fit_generator(generator.flow(X_train, Y_train,
                    batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size + 1,
                    nb_epoch=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(X_valid, Y_valid),
                    verbose=1)
