# load necessary modules
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from model import model_generate
from metrics_callback import MetricsCallback
from checkpoint_callback import CheckpointCallback
from earlystopping_callback import EarlyStoppingCallback
from learningRate_callback import LearningRateCallback
from keras import optimizers

# read the datasets
X_train = np.load('../../data/multitask_learning/Gender/Train/final_data.npy')
Y_train = np.load('../../data/multitask_learning/Gender/Train/final_labels_data.npy')

X_valid = np.load('../../data/multitask_learning/Gender/Valid/final_data.npy')
Y_valid = np.load('../../data/multitask_learning/Gender/Valid/final_labels_data.npy')

if __name__ == '__main__':

    # parameters
    batch = 256
    epochs = 10000
    verbose = 1
    input_shape = (64, 64, 1)
    patience = 310
    patience_lr = 60

    
    # callbacks
    metrics_callback = MetricsCallback(validation_data=(X_valid, Y_valid))
    checkpoint_callback = CheckpointCallback(monitor='accuracy', verbose=verbose, save_weights_only=False)
    early_stopping_callback = EarlyStoppingCallback(monitor='accuracy', min_delta=0, patience=patience, verbose=verbose)
    learning_rate_callback = LearningRateCallback(monitor='accuracy', min_delta=0, patience=patience_lr, verbose=verbose)

    callbacks = [metrics_callback, checkpoint_callback, early_stopping_callback, learning_rate_callback]

    # Generate the architecture
    model = model_generate(2)

    # Configure the learning process by compiling the network
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    print(model.summary())

    # Train the model for a fixed number of epochs
    model.fit(X_train, Y_train, batch_size=batch, verbose=verbose, callbacks=callbacks,
              validation_data=(X_test, Y_test), epochs=epochs)
    