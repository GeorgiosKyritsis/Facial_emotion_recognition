from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from collections import deque
import os
import csv
import warnings


class MetricsCallback(Callback):
    def __init__(self):
        self.filepath = os.path.join('../../../logs/VGG16_Finetuning_face_conv3_3', 'metrics.csv')

        with open(self.filepath, 'w', newline='') as newFile:
            new_file_writer = csv.writer(newFile)
            new_file_writer.writerow(['epoch', 'train_categorical_cross_entropy', 'val_categorical_cross_entropy'])


    def on_epoch_end(self, epoch, logs={}):
        #X_val = self.validation_data[0]
        #Y_val = self.validation_data[1]

        #Y_val_pred = self.model.predict(X_val)
        #Y_val_pred_arg = np.argmax(Y_val_pred, axis=1)

        #count = 0
        #for i in range(Y_val.shape[0]):
        #    if (Y_val[i][Y_val_pred_arg[i]] == np.max(Y_val[i])):
        #        count += 1

        #accuracy_score_val = count / Y_val.shape[0]

        train_categorical_cross_entropy = logs.get('loss')
        val_categorical_cross_entropy = logs.get('val_loss')

        with open(self.filepath, 'a', newline='') as newFile:
            new_file_writer = csv.writer(newFile)
            #new_file_writer.writerow([epoch, accuracy_score_val, log_loss_val])
            new_file_writer.writerow([epoch, train_categorical_cross_entropy, val_categorical_cross_entropy])


class CheckpointCallback1(Callback):

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            file = '../../../logs/VGG16_Finetuning_face_conv3_3/' + 'model-epoch-{0:06d}.hdf5'.format(epoch)
            self.model.save(file)
