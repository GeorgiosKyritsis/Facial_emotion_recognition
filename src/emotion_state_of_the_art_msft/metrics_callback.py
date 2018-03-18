# load necessary modules
from keras.callbacks import Callback
import numpy as np
import os
import csv

'''
Calculate the metric (accuracy) and the loss (categorical_cross_entropy) at every epoch
and save to file
'''

class MetricsCallback(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.filepath = os.path.join('../../logs/Emotion/', 'metrics.csv')

        with open(self.filepath, 'w') as newFile:
            new_file_writer = csv.writer(newFile)
            new_file_writer.writerow(['epoch', 'accuracy', 'categorical_cross_entropy'])

    def on_epoch_end(self, epoch, logs={}):
        X_val = self.validation_data[0]
        Y_val = self.validation_data[1]

        Y_val_pred = self.model.predict(X_val)
        Y_val_pred_arg = np.argmax(Y_val_pred, axis=1)

        count = 0
        for i in range(Y_val.shape[0]):
            if (Y_val[i][Y_val_pred_arg[i]] == np.max(Y_val[i])):
                count += 1

        accuracy_score_val = count / Y_val.shape[0]

        log_loss_val = logs.get('val_loss')

        with open(self.filepath, 'a') as newFile:
            new_file_writer = csv.writer(newFile)
            new_file_writer.writerow([epoch, accuracy_score_val, log_loss_val])
