import warnings
from keras.callbacks import Callback
import numpy as np
from collections import deque
import csv

'''
Check at the end of every epoch if the metric (accuracy) improves
If so save the weights or the entire model to the logs folder
'''

class CheckpointCallback(Callback):
    def __init__(self, monitor, verbose=0, save_weights_only=True, mode='auto'):

        # metric to monitor (accuracy)
        self.monitor = monitor
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        self.index = None
        # log file with the metric values at every epoch
        self.filepath = os.path.join('../../logs/Emotion/', 'metrics.csv')

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
    def on_train_begin(self, epoch, logs={}):

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    '''
    Check if accuracy improves
    If so save the model or the weights
    '''
    def on_epoch_end(self, epoch, logs={}):

        current = self._get_accuracy(self.filepath)
        if current is None:
            warnings.warn('Checkpoint requires {} available!'.format(self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            if self.save_weights_only:
                file = '../../logs/Emotion/' + 'weights-epoch-{0:06d}-acc_test-{1:.5f}.hdf5'.format(epoch, self.best)
                self.model.save_weights(file)
            else:
                file = '../../logs/Emotion/' + 'model-epoch-{0:06d}-acc_test-{1:.5f}.hdf5'.format(epoch, self.best)
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
