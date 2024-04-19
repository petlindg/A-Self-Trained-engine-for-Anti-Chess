import time
from copy import deepcopy

from Game.Utils import translate_moves_to_output
from chess.chessboard import Chessboard
from chess.utils import Color
from Game.Player import Player
from multiprocessing import Queue
from typing import List


class EvalGame:
    """
    A class representing one game of antichess given two queues linked to 2 different NeuralNetworkProcessEval instances through incoming and outgoing queues.
    """

    player : List[Player]

    def __init__(self,
                 initial_state: Chessboard,
                 outgoing_queue_1: Queue,
                 incoming_queue_1: Queue,
                 outgoing_queue_2: Queue,
                 incoming_queue_2: Queue,
                 uid: int):
        """
        A class representing one game of antichess given two queues linked to 2 different NeuralNetworkProcessEval instances through incoming and outgoing queues.

        :param initial_state: The Chessboard to play from
        :param outgoing_queue_1: The outgoing queue to contain prediction request to a NeuralNetworkProcessEval instance
        :param incoming_queue_1: The incoming queue to contain results from predictions from a NeuralNetworkProcessEval instance
        :param outgoing_queue_2: The outgoing queue to contain prediction request to a NeuralNetworkProcessEval instance
        :param incoming_queue_2: The incoming queue to contain results from predictions from a NeuralNetworkProcessEval instance
        """
        self.model_1 = Color(uid%2)
        self.model_2 = Color((uid+1)%2)
        
        self.uid = uid
        self.player = [None, None]
        self.player[self.model_1] = Player(deepcopy(initial_state), outgoing_queue_1, incoming_queue_1, uid)
        self.player[self.model_2] = Player(deepcopy(initial_state), outgoing_queue_2, incoming_queue_2, uid)
        self.player[1].get_next_move()


    
    def run(self):
        """
        Runs the game until fininshed.
        """    
        move_counter = 0
        current_player = move_counter%2

        while self.player[current_player].current_state.get_game_status()==3:
           move = self.player[current_player].get_next_move()
           self.player[0].update_tree(move)
           self.player[1].update_tree(move)

           current_player = (current_player+1)%2
           move_counter += 1

        result = self.player[0].current_state.get_game_status()
        if result == self.model_1:
            return ("model_1", move_counter)
        elif result == self.model_2:
            return ("model_2", move_counter)
        else:
            return ("draw", move_counter)
        

