import tempfile
import os

import h5py
import keras
from keras.models import load_model, save_model

from dlgo import agent
from dlgo.agent.predict import DeepLearningAgent
from dlgo.httpfrontend import get_web_app
from dlgo import encoders

def load_model_from_hdf5_group(f, custom_objects=None):
    # Extract the model into a temporary file.
    # Then we can use Keras load_model to read it.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    print("tempfname={}".format(tempfname))
    try:
        os.close(tempfd)
        serialized_model = h5py.File(tempfname, 'w')
        root_item = f.get('kerasmodel')
        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value
        for k in root_item.keys():
            f.copy(root_item.get(k), serialized_model, k)
        serialized_model.close()
        #return load_model(tempfname, custom_objects=custom_objects)
        return keras.layers.TFSMLayer(tempfname, call_endpoint="serving_default")
    finally:
        os.unlink(tempfname)


def load_prediction_agent(h5file):
    model = load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)


model_file = h5py.File("./agents/betago.h5", "r")
bot_from_file = load_prediction_agent(model_file)

web_app = get_web_app({'predict': bot_from_file})
web_app.run()

