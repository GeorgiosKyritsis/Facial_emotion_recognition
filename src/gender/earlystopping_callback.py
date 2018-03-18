from keras.callbacks import Callback
import numpy as np
from collections import deque
import csv
import os
import warnings

'''
Stop training when the monitored metric (accuracy)
has stopped improving
'''

class EarlyStoppingCallback(Callback):
    def __init__(self, monitor, min_delta=0, patience=0, verbose=0, mode='auto'):

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.index = None
        self.filepath = os.path.join('../../logs/Gender', 'metrics.csv')
        self.wait = 0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn()
            warnings.warn('EarlyStopping mode {} is unknown, fallback to auto mode.'.format(self.mode), RuntimeWarning)
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
            warnings.warn('Early stopping requires {} available!'.format(self.monitor), RuntimeWarning)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience - 1:
                if self.verbose > 0:
                    print('')
                    print('=============================================================')
                    print('Patience {0:02d}, epoch {0:03d}: early stopping'.format(self.wait, epoch))
                    print('=============================================================')
                    print('')
                self.model.stop_training = True
            self.wait += 1
            print('')
            print('*******')
            print('Wait:', self.wait)
            print('Best:', self.best)
            print('*******')
            print('')
