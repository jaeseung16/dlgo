from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D


def alphago_model(input_shape, is_policy_net=False, num_filters=192, first_kernel_size=5, other_kernel_size=3):
    model = Sequential()
    model.add(Conv2D(num_filters, first_kernel_size, input_shape=input_shape, padding='same', data_format='channels_last', activation='relu'))

    for i in range(2,12):
        model.add(Conv2D(num_filters, other_kernel_size, padding='same', data_format='channels_last', activation='relu'))

    if is_policy_net:
        # UserWarning: You are using a softmax over axis -1 of a tensor of shape (1, 19, 19, 1). This axis has size 1. The softmax operation will always return the value 1, which is likely not what you intended. Did you mean to use a sigmoid instead?
        #model.add(Conv2D(filters=1, kernel_size=1, padding='same', data_format='channels_last', activation='softmax'))
        model.add(Conv2D(filters=1, kernel_size=1, padding='same', data_format='channels_last', activation='sigmoid'))
        model.add(Flatten())
        return model
    else:
        model.add(Conv2D(num_filters, other_kernel_size, padding='same', data_format='channels_last', activation='relu'))
        model.add(Conv2D(filters=1, kernel_size=1, padding='same', data_format='channels_last', activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        return model
