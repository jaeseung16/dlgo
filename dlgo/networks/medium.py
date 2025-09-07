from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, ZeroPadding2D, Input


def layers(input_shape):
    return [
        Input(shape=input_shape),
        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((2, 2), data_format='channels_last'),
        Conv2D(64, (5, 5), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format='channels_last'),
        Conv2D(64, (3, 3), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format='channels_last'),
        Conv2D(64, (3, 3), data_format='channels_last'),
        Activation('relu'),

        ZeroPadding2D((1, 1), data_format='channels_last'),
        Conv2D(64, (3, 3), data_format='channels_last'),
        Activation('relu'),

        Flatten(),
        Dense(512),
        Activation('relu'),
    ]
