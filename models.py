import tensorflow as tf
from tensorflow.keras import layers, models

keras = tf.keras


def my_model(input_shape=None):
    '''
        Input: 
            input_shape: A sequence of integers. E.g: (32, 32, 3)
        Output:
            Return a model
    '''
    model = models.Sequential()

    # tf.keras.layers.Conv2D(filters, kernel_size,... )
    # tf.keras.layers.MaxPool2D(pool_size,stride=None,... ), if stride=None then stride=pool_size

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # tf.keras.layers.Dense(units, activation=None,... ) # units: the number of neurons in that layer

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model
