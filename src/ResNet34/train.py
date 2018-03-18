# load necessary modules 
import numpy as np
import os
import resnet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
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
img_rows, img_cols = 48, 48
# grayscale images
img_channels = 1
# number of emotions
nb_classes = 8

verbose = 1

# hyperparameters
nb_epoch = 100
batch_size = 128


# Set up the callbacks
metrics_callback = MetricsCallback(training_data=(X_train/255.0, Y_train), validation_data=(X_valid, Y_valid))
checkpoint_callback = CheckpointCallback(monitor='val_accuracy', verbose=verbose, save_weights_only=False)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

callbacks = [metrics_callback, checkpoint_callback, lr_reducer]


# set up the architecture
model = resnet.ResnetBuilder.build_resnet_34((img_channels, img_rows, img_cols), nb_classes)
# set up the optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam')

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
