# load necessary modules
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from model import model_generate
from keras import optimizers


# read the datasets
X_train = np.load('../../data/multitask_learning/Age/Train/final_data.npy')
Y_train = np.load('../../data/multitask_learning/Age/Train/final_labels_data.npy')

X_valid = np.load('../../data/multitask_learning/Age/Valid/final_data.npy')
Y_valid = np.load('../../data/multitask_learning/Age/Valid/final_labels_data.npy')


if __name__ == '__main__':

    # parameters
    batch = 256
    epochs = 10000
    verbose = 1
    input_shape = (64, 64, 1)
    
    # Generate the architecture
    model = model_generate(1)

    # Configure the learning process by compiling the network
    model.compile(optimizer='adam', loss='mean_absolute_error')
    
    print(model.summary())

    # callback to save the best model
    checkpointer = ModelCheckpoint(filepath='../../logs/Age/weights.hdf5', verbose=1, save_best_only=True)

    # Train the model for a fixed number of epochs
    model.fit(X_train, Y_train, batch_size=batch, verbose=verbose,
              validation_data=(X_test, Y_test), epochs=epochs, callbacks=[checkpointer])
