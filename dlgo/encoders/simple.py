import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move
from dlgo.gotypes import Player, Point


class SimpleEncoder(Encoder):
    def __init__(self, board_size):
        """
        Args:
            board_size, tuple of (width, height)
        """
        self.board_width, self.board_height = board_size
        # 0 - 3: black stones with 1, 2, 3, 4+ liberties
        # 4 - 7: white stones with 1, 2, 3, 4+ liberties
        # 8: black plays next
        # 9: white plays next
        # 10: move would be illegal due to ko
        self.num_planes = 11

    def name(self):
        return 'simple'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        if game_state.next_player == Player.black:
            board_tensor[..., 8] = np.ones((self.board_height, self.board_width))
        else:
            board_tensor[..., 9] = np.ones((self.board_height, self.board_width))
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player, Move.play(p)):
                        board_tensor[r][c][10] = 1
                else:
                    # if go_string.num_liberties < 1:
                    # print("nextPlayer={}, row={}, col={}, num_liberties={}, string_color={} stones={} liberties={}"
                    #       .format(game_state.next_player, r, c, go_string.num_liberties,
                    #               go_string.color, go_string.stones, go_string.liberties)
                    #       )
                    liberty_plane = min(4, go_string.num_liberties) - 1
                    if go_string.color == Player.white:
                        liberty_plane += 4
                    board_tensor[r][c][liberty_plane] = 1

        return board_tensor

    def encode_point(self, point):
        """Turn a board point into an integer index."""
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """Turn an integer index into a board point."""
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return self.board_height, self.board_width, self.num_planes


def create(board_size):
    return SimpleEncoder(board_size)
