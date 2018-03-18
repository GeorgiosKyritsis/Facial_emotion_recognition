from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from collections import deque
import os
import csv
import warnings

# Saving categorical cross entropy and accuracy
# for every epoch for both training and validation set
class MetricsCallback(Callback):
    def __init__(self):
        self.filepath = os.path.join('../../../logs/VGG16_Finetuning_conv4_3', 'metrics.csv')

        with open(self.filepath, 'w', newline='') as newFile:
            new_file_writer = csv.writer(newFile)
            new_file_writer.writerow(['epoch', 'train_categorical_cross_entropy', 'val_categorical_cross_entropy'])


    def on_epoch_end(self, epoch, logs={}):

        train_categorical_cross_entropy = logs.get('loss')
        val_categorical_cross_entropy = logs.get('val_loss')

        with open(self.filepath, 'a', newline='') as newFile:
            new_file_writer = csv.writer(newFile)
            new_file_writer.writerow([epoch, train_categorical_cross_entropy, val_categorical_cross_entropy])


# Save model weights when the accuracy improves
class CheckpointCallback1(Callback):

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            file = '../../../logs/VGG16_Finetuning_conv4_3/' + 'model-epoch-{0:06d}.hdf5'.format(epoch)
            self.model.save(file)
