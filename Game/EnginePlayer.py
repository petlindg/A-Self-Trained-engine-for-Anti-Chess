from chess import Chessboard
from keras import Model
from node import Node
from copy import deepcopy


class EnginePlayer:
    """
    A class representing a player
    """

    def __init__(self, initial_state: Chessboard, model: Model):
        self.mcts = Node(state=initial_state, p=1, model=model)

    def get_move(self):
        self.run_mcts()
        return max(self.mcts.children, key=lambda node: node.visits).move

    def update(self, move):
        try:
            self.mcts = self.mcts.update_tree(move)
        except:
            # TODO: Lots of bs because mcts might not have children
            new_state = deepcopy(self.mcts.state)
            new_state.move(move)
            self.mcts = Node(state=new_state, p=1, model=self.mcts.model)



    def run_mcts(self):
        self.mcts.run()
