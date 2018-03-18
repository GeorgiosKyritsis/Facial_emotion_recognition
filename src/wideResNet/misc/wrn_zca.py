import numpy as np
import os
from PIL import Image

import wide_residual_network as wrn

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from callbacks import *

from tqdm import tqdm

# Read files
folder_train = '../data/Train48_zca/'
folder_valid = '../data/Valid48_zca/'
folder_test = '../data/Test48_zca/'

files_train = os.listdir(folder_train)
files_valid = os.listdir(folder_valid)
files_test = os.listdir(folder_test)

images_train = [file for file in files_train if 'img' in file]
images_valid = [file for file in files_valid if 'img' in file]
images_test = [file for file in files_test if 'img' in file]

labels_train = [file for file in files_train if 'img' in file]
labels_valid = [file for file in files_valid if 'img' in file]
labels_test = [file for file in files_test if 'img' in file]

# Sanity check
assert len(images_train) == len(labels_train)
assert len(images_valid) == len(labels_valid)
assert len(images_test) == len(labels_test)

# Create numpy arrays X_train, Y_train
#X_train = [np.reshape(np.array(Image.open(folder_train + 'img' + str(i) + '.png')), (48,48,1)) for i in range(len(images_train))]
#X_train = np.concatenate([arr[np.newaxis] for arr in X_train])
#X_train = X_train.astype('float32')

X_train = [np.load(folder_train + 'img' + str(i) + '.npy') for i in tqdm(range(len(images_train)))]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train])

Y_train = [np.load(folder_train + 'lab' + str(i) + '.npy') for i in tqdm(range(len(labels_train)))]
Y_train = np.concatenate([arr[np.newaxis] for arr in Y_train])

# Create numpy arrays X_valid, Y_valid
#X_valid = [np.reshape(np.array(Image.open(folder_valid + 'img' + str(i) + '.png')), (48,48,1)) for i in range(len(images_valid))]
#X_valid = np.concatenate([arr[np.newaxis] for arr in X_valid])
#X_valid = X_valid.astype('float32')

X_valid = [np.load(folder_valid + 'img' + str(i) + '.npy') for i in tqdm(range(len(images_valid)))]
X_valid = np.concatenate([arr[np.newaxis] for arr in X_valid])

Y_valid = [np.load(folder_valid + 'lab' + str(i) + '.npy') for i in tqdm(range(len(labels_valid)))]
Y_valid = np.concatenate([arr[np.newaxis] for arr in Y_valid])

# Concatenate X_train, X_valid and Y_train, Y_valid
X_train = np.concatenate((X_train, X_valid), axis=0)
Y_train = np.concatenate((Y_train, Y_valid), axis=0)

# Create numpy arrays X_test, Y_test
#X_test = [np.reshape(np.array(Image.open(folder_test + 'img' + str(i) + '.png')), (48,48,1)) for i in range(len(images_test))]
#X_test = np.concatenate([arr[np.newaxis] for arr in X_test])
#X_test = X_test.astype('float32')
#X_test = X_test / 255.0

X_test = [np.load(folder_test + 'img' + str(i) + '.npy') for i in tqdm(range(len(images_test)))]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test])

Y_test = [np.load(folder_test + 'lab' + str(i) + '.npy') for i in tqdm(range(len(labels_test)))]
Y_test = np.concatenate([arr[np.newaxis] for arr in Y_test])

# Sanity check
assert X_train.shape[0] == Y_train.shape[0]
#assert X_valid.shape[0] == Y_valid.shape[0]
assert X_test.shape[0] == Y_test.shape[0]

print('===Final shape===')
print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('X_test:', X_test.shape)
print('Y_test:', Y_test.shape)


# Real time augmentation
generator = ImageDataGenerator(rotation_range=10,
								width_shift_range=5./48,
								height_shift_range=5./48,
								horizontal_flip=True)

generator.fit(X_train, seed=0, augment=True)

img_shape = (48,48,1)
verbose = 1

# hyperparameters
nb_epoch = 300
batch_size = 100
#lr_schedule = [100, 150]

'''
def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02
    else:
        return 0.004
'''

# Set up the callbacks
metrics_callback = MetricsCallback(validation_data=(X_test, Y_test))
checkpoint_callback = CheckpointCallback(monitor='accuracy', verbose=verbose, save_weights_only=False)
#lr_callback = LearningRateScheduler(schedule=schedule)

callbacks = [metrics_callback, checkpoint_callback]


#For WRN-28-10 N = (nb_layers-4)/6 = (28-4)/6 = 4, k=10  (maybe memory constraints!!!) 4.17%
#For WRN-16-8 N = (nb_layers-4)/6 = (16-4)/6 = 2, k=8 4.81%
#For WRN-40-4 N = (nb_layers-4)/6 = (16-4)/6 = 2, k=8 4.97%

model = wrn.create_wide_residual_network(img_shape, nb_classes=8, N=4, k=10, dropout=0.5)

# set the optimizer
sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#model.load_weights('../logs_nodrop_128_zca/model-epoch-000032-acc_test-0.86199.hdf5')

# Model Summary
print(model.summary())

model.fit_generator(generator.flow(X_train, Y_train,
                    batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size + 1,
                    nb_epoch=nb_epoch,
                    callbacks=callbacks,
                    validation_data=(X_test, Y_test),
                    verbose=1)
