from keras import Model

from MCTS import MCTS
from chess import Chessboard, Color, Move


class Player:
    """
    A class representing a player
    """

    def __init__(self, initial_state: Chessboard, model: Model, color: Color):
        self.current_state = initial_state
        self.color = color
        self.mcts = MCTS(root_state=initial_state, player=color, model=model)
        self.history = []

    def run_mcts(self):
        self.mcts.run()

    def update_tree(self, move: Move, chessboard: Chessboard):
        self.mcts.update_tree(move, chessboard)

    def get_time_predicted(self):
        return self.mcts.time_predicted

    def get_next_move(self):
        self.run_mcts()
        self.mcts.root_node.print_selectively(2)

        potential_nodes = self.mcts.root_node.children

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
        self.history.append((self.current_state, mcts_dist, self.color))

        return best_move
