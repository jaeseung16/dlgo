import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.goboard import Move
from dlgo.gotypes import Point


class BetaGoEncoder(Encoder):
    """7 plane encoder as used in betago"""
    def __init__(self, board_size):
        """
        Args:
            board_size: tuple of (width, height)
        """
        self.board_width, self.board_height = board_size
        # 0 - 2. out stone with 1, 2, 3+ liberties
        # 3 - 5. opponent stone with 1, 2, 3+ liberties
        # 6. move would be illegal due to ko
        self.num_planes = 7

    def name(self):
        return 'betago'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        base_plane = {
            game_state.next_player: 0,
            game_state.next_player.other: 3,
        }
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)

                if go_string is None:
                    if game_state.does_move_violate_ko(game_state.next_player, Move.play(p)):
                        board_tensor[r][c][6] = 1
                else:
                    liberty_plane = min(3, go_string.num_liberties) - 1
                    liberty_plane += base_plane[go_string.color]
                    board_tensor[r][c][liberty_plane] = 1

        return board_tensor

    def encode_point(self, point):
        """Turn a board point into an integer index."""
        # Points are 1-indexed
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        """Turn an integer index into a board point."""
        row = index // self.board_width
        col = index % self.board_height
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_width * self.board_height

    def shape(self):
        return (self.board_height, self.board_width, self.num_planes)


def create(board_size):
    return BetaGoEncoder(board_size)
