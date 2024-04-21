from keras import Model

from copy import deepcopy

from config import verbosity
from node import Node
from chess.chessboard import Chessboard
from chess.move import Move
from multiprocessing import Queue

class Player:
    """
    A class representing a player
    """

    def __init__(self, initial_state: Chessboard, outgoing_queue: Queue,
                 incoming_queue: Queue, uid: int):
        self.current_state = initial_state
        self.mcts = Node(state=initial_state, p=1, outgoing_queue=outgoing_queue, incoming_queue=incoming_queue, uid=uid)
        self.mcts.expand()
        self.mcts.add_noise(frac=1)
        self.history = []

    def run_mcts(self):
        self.mcts.run()

    def update_tree(self, move: Move):
        self.mcts = self.mcts.update_tree(move)

    def get_time_predicted(self):
        return self.mcts.time_predicted

    def get_next_move(self):

        self.run_mcts()
        if verbosity != 0:
            self.mcts.print_selectively(2)

        potential_nodes = self.mcts.children

        max_visits = 0
        best_move = None
        mcts_dist = []
        visit_total = 0

        for node in potential_nodes:
            visit_total += node.visits
            mcts_dist.append((node.visits, node.move))
            if node.visits > max_visits:
                max_visits = node.visits
                best_move = node.move

        # prepare the float distribution of all actions
        # so that the model can use it for backpropagation
        mcts_dist = [(n / visit_total, move) for (n, move) in mcts_dist]

        # add values to the game history recording all moves
        state_clone = deepcopy(self.current_state)
        self.history.append((state_clone, mcts_dist))

        return best_move
