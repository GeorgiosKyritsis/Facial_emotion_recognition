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
    def __init__(self, training_data, validation_data):
        self.training_data = training_data
        self.validation_data = validation_data
        self.filepath = os.path.join('../../logs/ResNet18', 'metrics.csv')

        with open(self.filepath, 'w', newline='') as newFile:
            new_file_writer = csv.writer(newFile)
            new_file_writer.writerow(['epoch', 'train_accuracy', 'val_accuracy', 'train_categorical_cross_entropy', 'val_categorical_cross_entropy'])

    def _get_accuracy(self, data):
        X = data[0]
        Y = data[1]

        Y_pred = self.model.predict(X)
        Y_pred = np.argmax(Y_pred, axis=1)

        count = 0
        for i in range(Y.shape[0]):
            if (Y[i][Y_pred[i]] == np.max(Y[i])):
                count += 1

        accuracy_score = count / Y.shape[0]

        return accuracy_score


    def on_epoch_end(self, epoch, logs={}):

        train_accuracy = self._get_accuracy(self.training_data)
        val_accuracy = self._get_accuracy(self.validation_data)
        train_categorical_cross_entropy = logs.get('loss')
        val_categorical_cross_entropy = logs.get('val_loss')

        with open(self.filepath, 'a', newline='') as newFile:
            new_file_writer = csv.writer(newFile)
            #new_file_writer.writerow([epoch, accuracy_score_val, log_loss_val])
            new_file_writer.writerow([epoch, train_accuracy, val_accuracy, train_categorical_cross_entropy, val_categorical_cross_entropy])


# Save model weights when the accuracy improves
class CheckpointCallback(Callback):
    def __init__(self, monitor, verbose=0, save_weights_only=False, mode='auto'):

        self.monitor = monitor
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        self.index = None
        self.filepath = os.path.join('../../logs/ResNet18', 'metrics.csv')

        if mode not in ['auto', 'min', 'max']:
            warnings.warn()
            warnings.warn('Checkpoint mode {} is unknown, fallback to auto mode.'.format(self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def _get_index(self, csv_filename):

        with open(csv_filename, newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)
        self.index = row1.index(self.monitor)

    def _get_accuracy(self, csv_filename):

        # get the index of the metric
        if self.index is None:
            self._get_index(csv_filename)
        with open(csv_filename, 'r') as f:
            try:
                lastrow = deque(csv.reader(f), 1)[0]
            except IndexError:
                lastrow = None
            return np.float32(lastrow[self.index])

    def on_train_begin(self, epoch, logs={}):

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs={}):

        current = self._get_accuracy(self.filepath)
        if current is None:
            warnings.warn('Checkpoint requires {} available!'.format(self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            if self.save_weights_only:
                file = '../../logs/ResNet18/' + 'weights-epoch-{0:06d}-acc_test-{1:.5f}.hdf5'.format(epoch, self.best)
                self.model.save_weights(file)
            else:
                file = '../../logs/ResNet18/' + 'model-epoch-{0:06d}-acc_test-{1:.5f}.hdf5'.format(epoch, self.best)
                self.model.save(file)

            if self.verbose:
                if self.save_weights_only:
                    print('')
                    print('=============================================================')
                    print('Metric improved. Saving weights to file:', file)
                    print('=============================================================')
                    print('')
                else:
                    print('')
                    print('=============================================================')
                    print('Metric improved. Saving model to file:', file)
                    print('=============================================================')
                    print('')
