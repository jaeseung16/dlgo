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

encoder = OnePlaneEncoder((go_board_rows, go_board_cols))

processor = GoDataProcessor(encoder=encoder.name())

generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test', num_games, use_generator=True)
#generator = processor.load_go_data('train', num_games)
#test_generator = processor.load_go_data('test', num_games)

input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
network_layers = small.layers(input_shape)
model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


epochs=5
batch_size = 128
model.fit(x=generator.generate(batch_size, num_classes),
          epochs=epochs,
          verbose=1,
          callbacks=[ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')],
          validation_data=test_generator.generate(batch_size, num_classes),
          steps_per_epoch=generator.get_num_samples() / batch_size,
          validation_steps=test_generator.get_num_samples() / batch_size
          )

model.evaluate(test_generator.generate(batch_size, num_classes),
               stpes=test_generator.get_num_samples() / batch_size)

if __name__ == '__main__':
    multiprocessing.freeze_support()