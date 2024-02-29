

from chess import Chessboard, Move, Color

class Node:
    def __init__(self):
        # general tree variables
        self.root_node
        self.parent
        self.children
        self.leaf
        self.terminal
        # node specific variables
        self.state :  Chessboard
        self.move : Move
        self.player : Color

        self.visits : int
        self.value : float
        self.p : float

    def __str__(self):
        pass


