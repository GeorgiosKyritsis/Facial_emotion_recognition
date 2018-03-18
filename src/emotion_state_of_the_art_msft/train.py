# load necessary modules
import numpy as np
from model import model_generate
from metrics_callback import MetricsCallback
from checkpoint_callback import CheckpointCallback
from earlystopping_callback import EarlyStoppingCallback
from learningRate_callback import LearningRateCallback
from keras import optimizers

# read the datasets
X_train = np.load('../../data/multitask_learning/Emotion/Train/final_data_aug.npy')
Y_train = np.load('../../data/multitask_learning/Emotion/Train/final_labels_data_aug.npy')

X_valid = np.load('../../data/multitask_learning/Emotion/Train/final_data.npy')
Y_valid = np.load('../../data/multitask_learning/Emotion/Train/final_labels_data.npy')

if __name__ == '__main__':

    # hyper-parameters
    batch = 256
    epochs = 10000
    verbose = 1
    input_shape = (64, 64, 1)
    num_classes = 8
    patience = 310
    patience_lr = 60
    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.9


    # callbacks
    metrics_callback = MetricsCallback(validation_data=(X_valid, Y_valid))
    checkpoint_callback = CheckpointCallback(monitor='accuracy', verbose=verbose, save_weights_only=True)
    early_stopping_callback = EarlyStoppingCallback(monitor='accuracy', min_delta=0, patience=patience, verbose=verbose)
    learning_rate_callback = LearningRateCallback(monitor='accuracy', min_delta=0, patience=patience_lr, verbose=verbose)

    callbacks = [metrics_callback, checkpoint_callback, early_stopping_callback, learning_rate_callback]

    # Generate the architecture
    model = model_generate(8)

    # Stochastic gradient descent optimizer
    sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    # Configure the learning process by compiling the network
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print(model.summary())

    # Train the model for a fixed number of epochs
    model.fit(X_train, Y_train, batch_size=batch, verbose=verbose, callbacks=callbacks,
              validation_data=(X_valid, Y_valid), epochs=epochs)
