import multiprocessing

from dlgo.data.parallel_processor import GoDataProcessor
#from dlgo.data.processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import small
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 100

print("Initializing encoder")
encoder = OnePlaneEncoder((go_board_rows, go_board_cols))

print("Initializing data processor")
processor = GoDataProcessor(encoder=encoder.name())

print(">>> Generator for training data")
generator = processor.load_go_data('train', num_games, use_generator=True)

print(">>> Generator for test data")
test_generator = processor.load_go_data('test', num_games, use_generator=True)
#generator = processor.load_go_data('train', num_games)
#test_generator = processor.load_go_data('test', num_games)

input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
print("input_shape={}".format(input_shape))
network_layers = small.layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

import datetime
now = datetime.datetime.now()
print("Starting at {}".format(now.strftime("%Y%m%d %H:%M:%S")))

model.summary()

epochs = 5
batch_size = 128
model.fit(x=generator.generate(batch_size, num_classes),
          epochs=epochs,
          verbose=1,
          callbacks=[ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')],
          validation_data=test_generator.generate(batch_size, num_classes),
          steps_per_epoch=generator.get_num_samples() / batch_size,
          validation_steps=test_generator.get_num_samples() / batch_size
          )

model.evaluate(x=test_generator.generate(batch_size, num_classes),
               steps=int(test_generator.get_num_samples() / batch_size))

now = datetime.datetime.now()
print("Finished at {}".format(now.strftime("%Y%m%d %H:%M:%S")))

if __name__ == '__main__':
    multiprocessing.freeze_support()