import time
from copy import deepcopy

from Game.Utils import translate_moves_to_output
from chess.chessboard import Chessboard
from chess.utils import Color
from Game.Player import Player
from multiprocessing import Queue
from typing import List
from node import Node

def sum_legal_p(node:Node):
    sum = node.p_legal
    for c in node.children:
        sum += sum_legal_p(c)
    return sum


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

        avg_legal_p = [[], []]

        while self.player[current_player].current_state.get_game_status()==3:
           move = self.player[current_player].get_next_move()
           avg_legal_p[current_player].append(sum_legal_p(self.player[current_player].mcts)/self.player[current_player].mcts.visits)
           self.player[0].update_tree(move)
           self.player[1].update_tree(move)

           current_player = (current_player+1)%2
           move_counter += 1

        model_1_p = sum(avg_legal_p[self.model_1])/len(avg_legal_p[self.model_1])
        model_2_p = sum(avg_legal_p[self.model_2])/len(avg_legal_p[self.model_2])

        result = self.player[0].current_state.get_game_status()
        if result == self.model_1:
            return ("model_1", move_counter, model_1_p, model_2_p)
        elif result == self.model_2:
            return ("model_2", move_counter, model_1_p, model_2_p)
        else:
            return ("draw", move_counter, model_1_p, model_2_p)
        

