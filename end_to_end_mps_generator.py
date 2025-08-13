import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

os.environ['NO_PROXY'] = '*'

import h5py

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large

go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols
encoder = SimpleEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name(), data_directory="/Volumes/WD14TB/dlgo/data")

# 179689 = 128 * 1403 + 105
num_games = 64000

print(">>> Generator for training data")
generator = processor.load_go_data('train', num_games, use_generator=True)

#print(">>> Generator for test data")
#test_generator = processor.load_go_data('test', num_games, use_generator=True)

input_shape = (go_board_rows, go_board_cols, encoder.num_planes)
model = Sequential()
network_layers = large.layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.summary()
epochs = 20
batch_size = 128
model.fit(x=generator.generate(batch_size, nb_classes),
          epochs=epochs,
          verbose=1,
          callbacks=[ModelCheckpoint('./checkpoints/simple_model_epoch_{epoch}.h5')],
          #validation_data=test_generator.generate(batch_size, nb_classes),
          steps_per_epoch=int(generator.get_num_samples() / batch_size),
          #validation_steps=int(test_generator.get_num_samples() / batch_size)
          )
deep_learning_bot = DeepLearningAgent(model, encoder)
deep_learning_bot.serialize(h5py.File("./agents/deep_bot_gpu_eleven.h5", "w"))

#model_file = h5py.File("../agents/deep_bot.h5", "r")
#bot_from_file = load_prediction_agent(model_file)
#
#web_app = get_web_app({'predict': bot_from_file})
#web_app.run()
