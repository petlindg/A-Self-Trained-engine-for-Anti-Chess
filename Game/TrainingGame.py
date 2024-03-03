import time

from Game.Utils import translate_moves_to_output
from chess import Chessboard, Color
from Game.Player import Player
from keras.models import Model


class TrainingGame:
    """
    A class representing one game of antichess
    """

    def __init__(self, initial_state: Chessboard, white_model: Model, black_model: Model = None):
        self.current_state = initial_state
        self.game_over = False
        self.game_history = []
        self.swap = False

        if black_model is None:
            black_model = white_model

        self.white = Player(initial_state, white_model, Color.WHITE)
        self.black = Player(initial_state, black_model, Color.BLACK)

    def game_ended(self):
        """Checks the status of the current Game

        :return: bool, true if the game has ended, otherwise false
        """

        return self.current_state.get_game_status() != 3

    def make_move(self, player: Color):
        """Method that performs a single move for a player, depending on which player it is
        it will perform the move by either using the p1_tree or the p2_tree. The game history is shared
        between the 2 players.

        :param player: Color, Player that performs a move
        :return: Float, time that the tree spent predicting.
        """

        current_player = self.white if (player == Color.WHITE) else self.black
        next_move = current_player.get_next_move()

        self.current_state.move(next_move)

        self.white.update_tree(next_move, self.current_state)
        self.black.update_tree(next_move, self.current_state)

        print(f'player: {player}', next_move)

        return current_player.get_time_predicted()

    def run(self):
        """Plays through a game until the game ends, performing moves between both players

        :return: list[Chessboard, MCTS_distribution, value], the training data generated from the game
        """

        start_time = time.time()
        predict_time = 0

        while not self.game_ended():
            print(self.current_state)
            print('player: ', self.current_state.player_to_move)
            time_predicted = self.make_move(player=self.current_state.player_to_move)
            predict_time += time_predicted

        end_time = time.time()
        total_time = end_time - start_time
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
        print(f'Time taken: {total_time} | Time Predicted: {predict_time} | % {predict_time / total_time * 100}')

        return winner

    def get_history(self):
        status = self.current_state.get_game_status()

        if status == 0:
            winner = Color.WHITE
        elif status == 1:
            winner = Color.BLACK
        else:
            winner = 'draw'

        finalized_history = []
        for (state, mcts, player) in (self.white.history + self.black.history):
            # if nobody won and it's a draw
            if winner == 'draw':
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0.5))
            # if winning player
            elif winner == player:
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 1))
            # if losing player
            elif winner is not player:
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0))

        return finalized_history
