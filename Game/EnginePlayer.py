from chess import Chessboard
from keras import Model
from node import Node
from multiprocessing import Queue


class EnginePlayer:
    """
    A class representing a player
    """

    def __init__(self, initial_state: Chessboard = None, model: Model = None, mcts: Node = None):
        if mcts is None:
            # Assuming 'initial_state' is your starting chessboard state and 'model' is your loaded neural network model
            self.mcts = Node(state=initial_state, p=1,model=model)

        else:
            self.mcts = mcts
    def run_mcts(self):
        self.mcts.run()

    def get_move(self):
       
        self.run_mcts()
        
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

        return best_move

    def update(self, move):
        
        self.mcts.state.move(move)
        self.mcts = Node(state=self.mcts.state, p=1, model=self.mcts.model)

    