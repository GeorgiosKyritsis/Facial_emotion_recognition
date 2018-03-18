from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D


def VGG(model_type='D', dropout=0.5, num_classes=1000, input_shape=(224, 224, 3)):
    """Implements the VGG model.
    
    The function implements the following 5 versions of the VGG model:
    'A': VGG model with 11 layers.
    'B': VGG model with 13 layers.
    'C': VGG model with 16 layers (with 1x1 convolutions)
    'D': VGG model with 16 layers
    'E': VGG model with 19 layers.
    
    Args:
        model_type (str): One of {'A', 'B', 'C', 'D', 'E'}. Defaults to 'D'.
            The type of model to implement.
        dropout (float): Parameter to the dropout layer (after the first two FC layers each). Defaults to 0.5
            If set to 'None', there will be no dropout layer.
        num_classes (int): The number of output classes. Defaults to 1000 (as in Imagenet).
            This will determine the number of neurons (units) in the last FC layer.
        input_shape (tuple): The shape of the input layer. Defaults to (224, 224, 3).
            The dimensions of the input tensor (of a single sample, i.e. do not include batch size).
            
            The tuple is of form (<image_height>, <image_width>, <channels>).
            The value of 'channels' could be 3(RGB) or 1(L).
    
    Returns:
        An instance (keras.models.Model): A keras model.
            The required VGG model is returned.
    
    References:
        - Very Deep Convolutional Networks for Large-Scale Image Recognition (https://arxiv.org/abs/1409.1556)
    """
    
    model_type = model_type.upper()
    assert model_type in ['A', 'B', 'C', 'D', 'E'], "Invalid value of 'model_type'\n" \
    "It should be one of {'A', 'B', 'C', 'D', 'E'}"
    
    assert (type(input_shape) == tuple), "Invalid value of 'input_shape'\n" \
    "It should be of form (<image_height>, <image_width>, <channels>)"
    
    assert (len(input_shape) == 3), "Invalid value of 'input_shape'\n" \
    "It should be of form (<image_height>, <image_width>, <channels>)"
    
    if(dropout != None):
        assert ((type(dropout) == float) and (0 <= dropout <= 1)), "Invalid value of 'dropout'.\n" \
        "It should be a real number between 0 and 1 (inclusive)."

    # Implementing the model
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu', input_shape=input_shape))
    if model_type in ['B', 'C', 'D', 'E']:
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    if model_type in ['B', 'C', 'D', 'E']:
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    if model_type in ['C']:
        model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                         activation='relu'))
    if model_type in ['D', 'E']:
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    if model_type in ['E']:
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    if model_type in ['C']:
        model.add(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                         activation='relu'))
    if model_type in ['D', 'E']:
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    if model_type in ['E']:
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                     activation='relu'))
    if model_type in ['C']:
        model.add(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                         activation='relu'))
    if model_type in ['D', 'E']:
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    if model_type in ['E']:
        model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                         activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    if(dropout != None):
        model.add(Dropout(dropout))
    model.add(Dense(units=4096, activation='relu'))
    if(dropout != None):
        model.add(Dropout(dropout))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model