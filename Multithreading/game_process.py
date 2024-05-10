import multiprocessing
import random

import config
from Game.game import Game
from Player.engine_player import EnginePlayer


class GameProcess(multiprocessing.Process):
    """
    Class representing a single process that plays games using the neural network. 
    This class communicates with and sends requests to the neural network process and recieves evaluations from the 
    neural network.
    """
    def __init__(self, initial_state, player_1: EnginePlayer, player_2: EnginePlayer = None):
        """
        The game process class has two queues, the incoming and the outcoming queue
        the incoming queue is the data that the game process recieves from the neural network
        and the outgoing queue is the queue that the game process puts data on, for the network process to handle
        the uid is the unique identifier for the process, it is used for identification purposes. 
        """
        super(GameProcess, self).__init__()
        self.player_1 = player_1
        self.player_2 = player_2
        self.initial_state = initial_state

    def run(self):
        """
        Function that continually plays games by sending requests to the neural network
        """
        from chess.chessboard import Chessboard
        from Game.TrainingGame import TrainingGame
        # TODO re-enable random states once that is on main
        #from Game.state_generator import generate_random_state

        random.seed()
        # while the process is running, keep running training games
        while True:
            self.chessboard = Chessboard(self.initial_state)

            random_state = None #generate_random_state(config.piece_list)
            if config.random_state_generation:
                game = Game(initial_state=Chessboard(random_state), player_1=self.player_1, player_2=self.player_2)
            else:
                game = Game(initial_state=self.chessboard, player_1=self.player_1, player_2=self.player_2)

            result = game.run()
            self.outgoing_queue.put(('finished', self.uid, (game.get_history(), result)))