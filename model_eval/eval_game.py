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
    A class representing one game of antichess
    """

    player : List[Player]

    def __init__(self, initial_state: Chessboard,
                 outgoing_queue_1: Queue,
                 incoming_queue_1: Queue,
                 outgoing_queue_2: Queue,
                 incoming_queue_2: Queue,
                 uid: int):
        self.player = []
        self.player.append(Player(deepcopy(initial_state), outgoing_queue_1, incoming_queue_1, uid))
        self.player.append(Player(deepcopy(initial_state), outgoing_queue_2, incoming_queue_2, uid))
        self.player[1].get_next_move()


    def run(self):
        
        move_counter = 0
        current_player = move_counter%2

        while self.player[current_player].current_state.get_game_status()==3:
           move = self.player[current_player].get_next_move()
           self.player[0].update_tree(move)
           self.player[1].update_tree(move)

           current_player = (current_player+1)%2
           move_counter += 1

        result = self.player[0].current_state.get_game_status()
        if result == 0:
            return ("white", move_counter)
        elif result == 1:
            return ("black", move_counter)
        else:
            return ("draw", move_counter)
        

