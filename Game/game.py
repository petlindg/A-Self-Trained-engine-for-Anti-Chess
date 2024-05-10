import time
from copy import deepcopy
from Game.Utils import translate_moves_to_output
from chess.chessboard import Chessboard
from chess.utils import Color
from Player.player import Player
from logger import Logger

class Game:
    """
    A class representing one game of antichess
    """

    def __init__(self, initial_state: Chessboard, player_1: Player, player_2: Player = None):
        self.logger = Logger("Game")
        self.current_state: Chessboard = deepcopy(initial_state)
        self.player_1: Player = player_1
        self.player_2: Player = player_2

    def game_ended(self):
        """Checks the status of the current Game

        :return: bool, true if the game has ended, otherwise false
        """

        return self.current_state.get_game_status() != 3

    def make_move(self):
        """Method that performs a single move for a player, depending on which player it is
        it will perform the move by either using the p1_tree or the p2_tree. The game history is shared
        between the 2 players.

        :param player: Color, Player that performs a move
        :return: Float, time that the tree spent predicting.
        """

        current_player = self.player_1
        if self.current_state.player_to_move == Color.BLACK and self.player_2 is not None:
            current_player = self.player_2

        next_move = current_player.get_next_move()

        self.current_state.move(next_move)
        self.logger.info(f"Player {self.current_state.player_to_move} makes move {next_move}")

        self.player_1.update_tree(next_move)
        if self.player_2 is not None:
            self.player_2.update_tree(next_move)

        return current_player.get_time_predicted()

    def run(self):
        """Plays through a game until the game ends, performing moves between both players

        :return: list[Chessboard, MCTS_distribution, value], the training data generated from the game
        """
        self.logger.info("Game start")
        start_time = time.time()
        predict_time = 0

        while not self.game_ended():
            self.logger.info(str(self.current_state))
            self.logger.info(f'player: {self.current_state.player_to_move}')
            time_predicted = self.make_move()
            predict_time += time_predicted

        end_time = time.time()
        total_time = end_time - start_time
        status = self.current_state.get_game_status()

        self.logger.info('===============================')
        if status == 0:
            self.logger.info("               White wins")
            # print('           white wins')
            winner = Color.WHITE
        elif status == 1:
            winner = Color.BLACK
            self.logger.info("Black wins")
            # print('           black wins')
        else:
            winner = 'draw'
            self.logger.info("draw")
            # print('             draw')
        self.logger.info("=========================")
        # print('===============================')
        # print(f'Time taken: {total_time} | Time Predicted: {predict_time} | % {predict_time / 1 * 100}')
        self.logger.info(f'Time taken: {total_time} | Time Predicted: {predict_time} | % {predict_time / 1 * 100}')

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
        for (state, mcts) in self.player_1.history:
            # if nobody won and it's a draw
            if winner == 'draw':
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0.5))
            # if winning player
            elif winner == state.player_to_move:
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 1))
            # if losing player
            elif winner is not state.player_to_move:
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0))

        return finalized_history
