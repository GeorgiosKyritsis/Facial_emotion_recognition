# load necessary modules
from keras.callbacks import Callback
import numpy as np
import keras.backend as K
import os
import warnings
import csv
from collections import deque

'''
Learning rate scheduler
Check if accuracy improves
If not reduce the learning rate over time
'''

class LearningRateCallback(Callback):

    def __init__(self, monitor, min_delta=0, patience=0, verbose=0, mode='auto'):
        self.schedule = 0.007
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.index = None
        self.filepath = os.path.join('../../logs/multitask_learning/Emotion/', 'metrics.csv')
        self.wait = 0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn()
            warnings.warn('Learning rate scheduler mode {} is unknown, fallback to auto mode.'.format(self.mode), RuntimeWarning)
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
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    # get the row of the last line of a csv file
    def _get_index(self, csv_filename):

        with open(csv_filename, newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)
        self.index = row1.index(self.monitor)


    '''
    Get the accuracy of the previous epoch
    in order to compare with the current accuracy
    ''' 
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

    # Necessary initialization at the beginning of the training
    def on_train_begin(self, logs={}):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    '''
    Check if accuracy improves
    If not count the number of epochs with no improvement
    after which training will be stopped 
    '''
    def on_epoch_end(self, epoch, logs={}):
        current = self._get_accuracy(self.filepath)
        if current is None:
            warnings.warn('Learning rate scheduler requires {} available!'.format(self.monitor), RuntimeWarning)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience - 1:
                if self.verbose > 0:
                    lr = np.float32(self.schedule)
                    print('')
                    print('=============================================================')
                    print('Learning rate changed to: {0:5f}'.format(lr))
                    print('=============================================================')
                    print('')
                    K.set_value(self.model.optimizer.lr, lr)
                    self.wait = 0
                    self.schedule /= 1.15
            self.wait += 1
