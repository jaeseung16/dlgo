from dlgo.encoders.base import Encoder
from dlgo.encoders.utils import is_ladder_escape, is_ladder_capture
from dlgo.gotypes import Point, Player
from dlgo.goboard_fast import Move
from dlgo.agent.helpers_fast import is_point_an_eye
import numpy as np

"""
Feature name            num of planes   Description
Stone colour            3               Player stone / opponent stone /empty
Ones                    1               A constant plane filled with 1
Zeros                   1               A constant plane filled with 0
Sensibleness            1               Whether a move is legal and does not fill its own eyes
Turns since             8               How many turns since a move was played
Liberties               8               Number of liberties (empty adjacent points)
Liberties after move    8               Number of liberties after this move is played
Capture size            8               How many opponent stones would be captured
Self-atari size         8               How many of own stones would be captured
Ladder capture          1               Whether a move at this point is a successful ladder capture
Ladder escape           1               Whether a move at this point is a successful ladder escape
"""

FEATURE_OFFSET = {
    "stone_color": 0,
    "ones": 3,
    "zeros": 4,
    "sensibleness": 5,
    "turns_since": 6,
    "liberties": 14,
    "liberties_after": 22,
    "capture_size": 30,
    "self_atari_size": 38,
    "ladder_capture": 46,
    "ladder_escape": 47,
    "current_player_color": 48
}


def offset(feature):
    return FEATURE_OFFSET[feature]


class AlphaGoEncoder(Encoder):
    def __init__(self, board_size=(19,19), use_player_plane=True):
        self.board_width, self.board_height = board_size
        self.use_player_plane = use_player_plane
        self.num_planes = 48 + use_player_plane

    def name(self):
        return 'alphago'

    def encode(self, game_state):
        board_tensor = np.zeros((self.board_height, self.board_width, self.num_planes))
        for r in range(self.board_width):
            for c in range(self.board_width):
                point = Point(row=r+1, col=c+1)

                go_string = game_state.board.get_go_string(point)
                if go_string and go_string.color == game_state.next_player:
                    board_tensor[r][c][offset("stone_color")]=1
                elif go_string and go_string.color == game_state.next_player.other:
                    board_tensor[r][c][offset("stone_color")+1] = 1
                else:
                    board_tensor[r][c][offset("stone_color")+2] = 1

                # TODO
                board_tensor[:, :, offset("ones")] = self.ones()
                board_tensor[:, :, offset("zeros")] = self.zeros()

                if not is_point_an_eye(game_state.board, point, game_state.next_player):
                    board_tensor[r][c][offset("sensibleness")] = 1

                ages = min(game_state.board.move_ages.get(r, c), 8)
                if ages > 0:
                    # print(ages)
                    board_tensor[r][c][offset("turns_since") + ages] = 1

                if game_state.board.get_go_string(point):
                    liberties = min(game_state.board.get_go_string(point).num_liberties, 8)
                    board_tensor[r][c][offset("liberties") + liberties] = 1

                move = Move(point)
                if game_state.is_valid_move(move):
                    new_state = game_state.apply_move(move)
                    liberties = min(new_state.board.get_go_string(point).num_liberties, 8)
                    board_tensor[r][c][offset("liberties_after") + liberties] = 1

                    adjacent_strings = [game_state.board.get_go_string(nb) for nb in point.neighbors()]

                    capture_count = 0
                    for go_string in adjacent_strings:
                        other_player = game_state.next_player.other
                        if go_string and go_string.num_liberties == 1 and go_string.color == other_player:
                            capture_count += len(go_string.stones)
                    capture_count = min(capture_count, 8)
                    board_tensor[r][c][offset("capture_size") + capture_count] = 1

                if go_string and go_string.num_liberties == 1:
                    go_string = game_state.board.get_go_string(point)
                    if go_string:
                        num_atari_stones = min(len(go_string.stones), 8)
                        board_tensor[r][c][offset("self_atari_size") + num_atari_stones] = 1

                if is_ladder_capture(game_state, point):
                    board_tensor[r][c][offset("ladder_capture")] = 1

                if is_ladder_escape(game_state, point):
                    board_tensor[r][c][offset("ladder_escape")] = 1

                if self.use_player_plane:
                    if game_state.next_player == Player.black:
                        board_tensor[:, :, offset("ones")] = self.ones()
                    else:
                        board_tensor[:, :, offset("zeros")] = self.zeros()

        return board_tensor

    def ones(self):
        return np.ones((self.board_height, self.board_width))

    def zeros(self):
        return np.zeros((self.board_height, self.board_width))

    def capture_size(self, game_state, num_planes=8):
        pass

    def encode_point(self, point):
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row+1, col=col+1)

    def num_points(self):
        return self.board_width + self.board_height

    def shape(self):
        return self.board_height, self.board_width, self.num_planes


def create(board_size):
    return AlphaGoEncoder(board_size)