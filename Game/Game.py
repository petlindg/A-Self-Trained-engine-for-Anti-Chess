from copy import deepcopy

from chess import Chessboard, Color
from CliPlayer import CliPlayer

class Game:
    """
    A class representing one game of antichess
    """

    def __init__(self, initial_state: Chessboard, white: CliPlayer, black: CliPlayer):
        self.current_state = deepcopy(initial_state)
        self.white = white
        self.black = black

    def game_ended(self):
        """Checks the status of the current Game
        :return: bool, true if the game has ended, otherwise false
        """
        return self.current_state.get_game_status() != 3

    def run(self):
        """Plays through a game until the game ends, performing moves between both players
        """
        while not self.game_ended():
            print('player: ', self.current_state.player_to_move)

            if self.current_state.player_to_move == Color.WHITE:
                next_move = self.white.get_next_move()
                self.black.opponent_move(next_move)
            else:
                next_move = self.black.get_next_move()
                self.white.opponent_move(next_move)

        status = self.current_state.get_game_status()
        print('===============================')
        if status == 0:
            print('           white wins')
            winner = Color.WHITE
        elif status == 1:
            winner = Color.BLACK
            print('           black wins')
        else:
            winner = 'draw'
            print('             draw')
        print('===============================')

        return winner