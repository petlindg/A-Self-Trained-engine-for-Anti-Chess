from chess import Chessboard
from keras import Model
from node import Node


class EnginePlayer:
    """
    A class representing a player
    """

    def __init__(self, initial_state: Chessboard = None, model: Model = None, mcts: Node = None):
        if mcts is None:
            self.mcts = Node(state=initial_state, p=1, model=model)
        else:
            self.mcts = mcts

    def get_move(self):
        self.mcts.run()
        self.mcts.print_selectively(2)
        return max(self.mcts.children, key=lambda node: node.visits).move

    def update(self, move):
        if move in map(lambda node: node.move, self.mcts.children):
            self.mcts = self.mcts.update_tree(move)
        else:
            self.mcts.state.move(move)
            self.mcts = Node(state=self.mcts.state, p=1, model=self.mcts.model)
