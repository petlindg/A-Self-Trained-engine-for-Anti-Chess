from copy import deepcopy
import random
from math import sqrt

import numpy as np
import time

from chess.chessboard import Chessboard
from chess.move import Move, calc_move
from config import exploration_constant, evaluation_method

from math import sqrt
from config import tree_iterations, exploration_constant, output_representation
from keras.models import Model

from nn_architecture import NeuralNetwork, OUTPUT_SHAPE, INPUT_SHAPE
from logger import Logger

from node import Node

c_hit = 0
c_miss = 0

logger = Logger("TrainingGame")
    
def fetch_p_from_move(move: Move, model_output: np.array):
    """Fetches the P value from the output array of the model

    :param move: Move, move to fetch the P value for
    :param model_output: np.array, array of the model's policy output
    :return: Float, the P value for the move
    """
    src_col = int(move.src_index % 8)
    src_row = int(move.src_index // 8)

    move_type = calc_move(move.src_index, move.dst_index, move.promotion_type)
    return model_output[0][0][src_row][src_col][move_type]

class MainNode:
    """
    Class for the MCTS tree and nodes of the mcts tree.
    """
    def __init__(self,
                 state: Chessboard,
                 v: float = None,
                 p: float = 1,
                 move: Move = None,
                 model: Model = None):

        # general tree variables
        self.children: list[Node] = []
        # node specific variables
        self.state: Chessboard = state
        self.move: Move = move
        self.v = v
        self.p: float = p
        # network
        self.model = model

    def __str__(self):
        """Method to return a node as a string.

        :return: String, A string representing the entire subtree from this node.
        """
        string_buffer = []
        self.print_tree(string_buffer, "", "")
        return "".join(string_buffer)

    def expand(self, node:Node):
        if self.children:
            global c_hit
            c_hit+=1
            self.expand_old(node)
            return self.v
        else:
            global c_miss
            c_miss+=1
            self.expand_self()
            self.expand_old(node)
            return self.v
        
    def expand_old(self, node:Node):
        for c in self.children:
            node.children.append(   
                Node(
                    state=c.state,
                    main_node=c,
                    p=c.p,
                    parent=node,
                    move=c.move
                )
            )

    def expand_self(self):
        """
        Performs an expansion on a leaf node, returning v of the node by the network
        and adding children to it

        :return: Value: float of the node expanded, as given by the network model
        """
        status = self.state.get_game_status()
        if status == 2:
            self.v = 0.5
        elif status == 0 or status == 1:
            self.v = 1
        else:
            if self.model:
                p_vector, self.v = self.possible_moves()
                for (move, p) in p_vector:
                    self.children.append(
                        MainNode(
                            state=self.state,
                            p=p,
                            move=move,
                            model=self.model
                        )
                    )
            else:
                moves = self.state.get_moves()
                if evaluation_method == 'dirichlet':
                    p_vals = np.random.dirichlet([1]*(len(moves)))
                    self.v = random.random()
                else:
                    p_vals = [1/len(moves)]*(len(moves))
                    self.v = 0.5

                for p, m in zip(p_vals, moves):
                    self.children.append(
                        MainNode(
                            state=self.state,
                            p=p,
                            move=m,
                            model=self.model
                        )
                    )

    def possible_moves(self):
        """Calculates all possible moves for a given chessboard using the neural network, and returns
           it as a list of tuples.

        :param state: Chessboard, the input state to calculate moves from
        :return: (list[Chessboard, Move, float], float), returns a list of new chessboards, the move
                 that was taken to get there and the float value P for that new state. In addition to this,
                 it also returns another float which is the value V from the neural network for the input state.
        """
        input_repr = self.state.translate_board()
        moves = self.state.get_moves()

        p, v = self.model.predict(input_repr, verbose=None)
        v = v[0][0]
        p_array = p.reshape(output_representation)
        return_list = []

        p_sum = 0
        for move in moves:
            p_val = fetch_p_from_move(move, p_array)
            p_sum += p_val
            return_list.append((move, p_val))

        # normalize the P values in the return list
        return_list = [(move, p_val/p_sum) for (move, p_val) in return_list]

        return return_list, v

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
            p = round(self.p, 5)
            val = self.v

            info_text = f'(p:{p}|v:{val}|move:{self.move})'
            string_buffer.append(info_text)
            string_buffer.append('\n')

            self.children.sort(key=lambda x: x.p, reverse=True)

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
        logger.info(f"\n{''.join(string_buffer)}")