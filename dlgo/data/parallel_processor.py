import os
import glob
import os.path
import tarfile
import gzip
import shutil
import numpy as np
import multiprocessing
import sys
from keras.utils import to_categorical
from numpy.random import SFC64

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler
from dlgo.data.generator import DataGenerator
from dlgo.encoders.base import get_encoder_by_name


def worker(jobinfo):
    try:
        clazz, encoder, data_dir, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(data_dir, zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GoDataProcessor:
    def __init__(self, encoder='simple', data_directory='data'):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory

    def load_go_data(self, data_type='train', num_samples=1000, use_generator=False):
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()

        print("Initializing Sampler: data_type={}, data_dir={}".format(data_type, self.data_dir))
        sampler = Sampler(data_dir=self.data_dir)
        data = sampler.draw_data(data_type, num_samples)

        # Map workload to CPUs
        self.map_to_workers(data_type, data)
        if use_generator:
            generator = DataGenerator(self.data_dir, data_type, data)
            # Either return a Go data generator...
            return generator
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            # ... or return consolidated data as before
            return features_and_labels

    def unzip_data(self, data_dir, zip_file_name):
        this_gz = gzip.open(data_dir + '/' + zip_file_name)

        tar_file = zip_file_name[0:-3]
        this_tar = open(data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    def process_zip(self, data_dir, zip_file_name, data_file_name, game_list):
        #print("data_dir={}, zip_file_name={}".format(data_dir, zip_file_name))
        tar_file = self.unzip_data(data_dir, zip_file_name)
        zip_file = tarfile.open(data_dir + '/' + tar_file)
        name_list = zip_file.getnames()
        total_examples = self.num_total_examples(zip_file, game_list, name_list)

        print("zip_file_name={}: total_examples={}".format(zip_file_name, total_examples))

        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))

        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)

            game_state, first_move_done = self.get_handicap(sgf)

            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                    if first_move_done and point is not None:
                        encoded_features = self.encoder.encode(game_state)
                        features[counter] = encoded_features
                        encoded_point = self.encoder.encode_point(point)
                        labels[counter] = encoded_point

                        #print("counter={}, encoded_features={}, features={}".format(counter, np.count_nonzero(encoded_features[..., 8]), np.count_nonzero(features[counter, ..., 8])))

                        #print("counter={}, encoded_features={}, features={}".format(counter, np.count_nonzero(encoded_features), np.count_nonzero(features[counter])))
                        #print("counter={}, point={}, label={}".format(counter, encoded_point, labels[counter]))
                        counter += 1

                    game_state = game_state.apply_move(move)
                    first_move_done = True

        print("zip_file_name={}: counter={}".format(zip_file_name, counter))

        feature_file_base = data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = data_dir + '/' + data_file_name + '_labels_%d'

        features = features[:counter, ...]
        labels = labels[:counter]

        chunk = 0
        # Due to files with large content, split up after chunksize
        chunksize = 1024
        while features.shape[0] >= chunksize:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)

    def consolidate_games(self, name, samples):
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + name
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), 19 * 19)
                feature_list.append(x)
                label_list.append(y)

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        feature_file = self.data_dir + '/' + name + '_features'
        label_file = self.data_dir + '/' + name + '_labels'

        np.save(feature_file, features)
        np.save(label_file, labels)

        print("features: file={}, shape={}".format(feature_file, features.shape))
        print("labels: file={}, shape={}".format(label_file, labels.shape))

        return features, labels

    @staticmethod
    def get_handicap(sgf):
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def map_to_workers(self, data_type, samples):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        #print(indices_by_zip_name)
        #print(zip_names)
        zips_to_process = []
        for zip_name in zip_names:
            #year = zip_name.split('-')
            #if not year == '2015' or not year == '2016':
            #    continue
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir + '/' + data_file_name):
                zips_to_process.append((self.__class__, self.encoder_string, self.data_dir, zip_name, data_file_name, indices_by_zip_name[zip_name]))

        cores = multiprocessing.cpu_count()
        print("multiprocessing.cpu_count()={} / # of cores={}".format(multiprocessing.cpu_count(), cores))
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('fork')
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(worker, zips_to_process)
        try:
            _ = p.get()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples

if __name__ == '__main__':
    multiprocessing.freeze_support()