from keras.layers import LeakyReLU
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D((3, 3), input_shape=input_shape, data_format='channels_last'),
        Conv2D(64, (7,7), padding='valid', data_format='channels_last'),
        LeakyReLU(),

        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        LeakyReLU(),

        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        LeakyReLU(),

        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(48, (5, 5), data_format='channels_last'),
        LeakyReLU(),

        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(48, (5, 5), data_format='channels_last'),
        LeakyReLU(),

        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(32, (5, 5), data_format='channels_last'),
        LeakyReLU(),

        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_last'),
        Conv2D(32, (5, 5), data_format='channels_last'),
        LeakyReLU(),

        Flatten(),
        Dense(1024),
        LeakyReLU(),
    ]
