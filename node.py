from math import sqrt

import numpy as np

from chess import Chessboard, Move, Color
from config import exploration_constant


class Node:
    def __init__(self, state: Chessboard,
                 p: float, player: Color,
                 parent, root_node,
                 root: bool = False,
                 move: Move = False):

        # general tree variables
        self.root_node: Node = root_node
        self.parent: Node = parent
        self.children: list[Node] = []
        self.root: bool = root
        self.leaf: bool = True
        self.terminal: bool = False
        # node specific variables
        self.state:  Chessboard = state
        self.move: Move = move
        self.player: Color = player
        self.v = 0
        self.visits: int = 0
        self.value: float = 0
        self.p: float = p
        # if the node is terminal, the end state determines which player won or lost or if it's a draw
        self.end_state = None

    def ucb(self, inverted: bool):
        if self.parent is not None:
            if not inverted:
                return (self.p * exploration_constant * sqrt(self.parent.visits) / (1 + self.visits)
                        + (self.value / self.visits if self.visits > 0 else 0))

            if inverted:
                return (self.p * exploration_constant * sqrt(self.parent.visits) / (1 + self.visits)
                        + ((1 - self.value) / self.visits if self.visits > 0 else 0))

    def __str__(self):
        """Method to return a node as a string.

        :return: String, A string representing the entire subtree from this node.
        """
        string_buffer = []
        self.print_tree(string_buffer, "", "")
        return "".join(string_buffer)

    def print_tree(self, string_buffer, prefix, child_prefix, depth=None):
        """Method that will iterate through the nodes and append strings onto the string_buffer list.

        :param string_buffer: List, list of strings that builds up over time
        :param prefix: String, a prefix string to the current node's values
        :param child_prefix: String, a prefix string for the new children
        :param depth: Integer, How deep to go from the current level
        :return: None
        """
        if depth is None or depth > 0:
            string_buffer.append(prefix)
            p = round(self.p, 10)
            val = round(self.value, 10)
            # v = round(self.v, 10)
            visits = self.visits
            if visits != 0:
                info_text = f'(p:{p}|v:{val}|n:{visits}|wr:{val/visits}|u:{self.ucb(False)}|move:{self.move})'
            else:
                info_text = f'(p:{p}|v:{val}|n:{visits}|wr:-|u:{self.ucb(False)}|move:{self.move})'
            string_buffer.append(info_text)
            string_buffer.append('\n')

            for i in range(0, len(self.children)):
                if i == len(self.children)-1:
                    self.children[i].print_tree(string_buffer,
                                                child_prefix + "└── ", child_prefix + "    ",
                                                (depth - 1 if depth else None))
                else:
                    self.children[i].print_tree(string_buffer,
                                                child_prefix + "├── ", child_prefix + "│   ",
                                                (depth - 1 if depth else None))

    def print_selectively(self, depth):
        """Method to print the subtree from the node with a given depth from the current node.

        :param depth: Integer, parameter to define how deeply to print

        :return: String, string representing the limited subtree from this node
        """
        string_buffer = []
        self.print_tree(string_buffer, "", "", depth)
        print("".join(string_buffer))

    def expand(self, new_states: list[(Chessboard, Move, float)], v: float):
        """
        Performs an expansion on a leaf node, assigning v to the leaf node
        and adding children to it

        :param new_states: list[(Chessboard, Move, float)], list of new states/children
        :param v: float, Neural Network value for this leaf node, indicating how good the network
        thinks the node is.

        :return: None
        """

        self.leaf = False
        self.v = v
        if self.player == Color.WHITE:
            next_player = Color.BLACK
        else:
            next_player = Color.WHITE

        for (state, move, p) in new_states:
            self.children.append(
                Node(
                    state=state,
                    move=move,
                    p=p,
                    parent=self,
                    player=next_player,
                    root_node=self.root_node
                )
            )
        # if the expansion failed to find any viable children
        if len(self.children) == 0:
            self.terminal = True

        # if we are expanding the root node for the first time, we add noise to the children
        if self.root:
            self.add_noise()

        self.backpropagate(self, v)

    def backpropagate(self, node, v: float):
        """
        Method that backpropagates the value v for the current node.

        :param node: Node class, the current node to backpropagate from
        :param v: Float, the value to backpropagate up the tree, v ranges from 0 to 1 (sigmoid)
        :return: None
        """

        node.value += v
        node.visits += 1
        
        # if we are at root node, we're done
        if node.parent == None:
            return

        # invert v value to because of color change before backpropagating
        self.backpropagate(node.parent, 1-v)



    def add_noise(self, dir_a=0.03, frac=0.25):
        """Adds dirichlet noise to all the p values for the children of the current node

        :param dir_a: Float, dirichlet alpha value. 0.03 is default
        :param frac: Float, fraction deciding how to weigh the P vs the noise, 0.25 is default
        :return:None
        """
        diri_dist = np.random.dirichlet([dir_a] * (len(self.children)))
        child_dist = [child.p for child in self.children]
        new_dist = [(1-frac)*p + frac*d for (p, d) in zip(child_dist, diri_dist)]
        # re normalizing the p values
        sum_dist = sum(new_dist)
        new_dist = [p/sum_dist for p in new_dist]
        for i, child in enumerate(self.children):
            child.p = new_dist[i]

    def select_child(self, move):
        for child in self.children:
            if child.move == move:
                return child
