from copy import deepcopy

from Model.model import Model
from Player.player import Player
from MCTS.node import Node
from chess.chessboard import Chessboard
from chess.move import Move

class EnginePlayer(Player):
    """
    A class representing a player
    """

    def __init__(self, initial_state: Chessboard, model: Model):
        self.current_state = initial_state
        self.mcts = Node(state=initial_state, p=1, model=model)
        self.mcts.expand()
        self.mcts.add_noise()
        self.history = []

    def run_mcts(self):
        self.mcts.run()

    def update_tree(self, move: Move):
        self.mcts = self.mcts.update_tree(move)

    def get_time_predicted(self):
        return self.mcts.time_predicted

    def print_tree(self):
        self.mcts.print_selectively(100)

    def get_next_move(self):

        print("Running MCTS")
        self.run_mcts()
        print("MCTS completed")
        # if verbosity != 0:
        #    self.mcts.print_selectively(2)

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
