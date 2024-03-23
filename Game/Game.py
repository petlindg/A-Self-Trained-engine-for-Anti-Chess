from copy import deepcopy
from chess import Chessboard, Color

class Game:
    """
    A class representing one game of antichess
    """

    def __init__(self, initial_state: Chessboard, white, black):
        self.chessboard = initial_state
        self.white = white
        self.black = black

    def game_ended(self):
        """Checks the status of the current Game
        :return: bool, true if the game has ended, otherwise false
        """
        return self.chessboard.get_game_status() != 3

    def run(self):
        """Plays through a game until the game ends, performing moves between both players
        """

        while not self.game_ended():
            if self.chessboard.player_to_move == Color.WHITE:
                next_move = self.white.get_move()
            else:
                next_move = self.black.get_move()

            if self.chessboard.is_valid_move(next_move):
                self.chessboard.move(next_move)
                self.white.update(next_move)
                self.black.update(next_move)

        return self.chessboard.get_game_status()