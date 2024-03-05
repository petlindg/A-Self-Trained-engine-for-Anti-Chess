from math import sqrt

import numpy as np
import time

from chess import Chessboard, Move, Color
from config import exploration_constant

global select_counter
global backpropagate_counter
select_counter = 0
backpropagate_counter = 0

import copy
from math import sqrt
import chess
from config import tree_iterations, exploration_constant, output_representation
from keras.models import Model

from nn_architecture import NeuralNetwork, OUTPUT_SHAPE, INPUT_SHAPE
    
def fetch_p_from_move(move: Move, model_output: np.array):
    """Fetches the P value from the output array of the model

    :param move: Move, move to fetch the P value for
    :param model_output: np.array, array of the model's policy output
    :return: Float, the P value for the move
    """
    src_col = int(move.src_index % 8)
    src_row = int(move.src_index // 8)

    move_type = chess.calc_move(move.src_index, move.dst_index, move.promotion_type)
    return model_output[0][0][src_row][src_col][move_type]

class Node:
    def __init__(self,
                 state: Chessboard,
                 p: float,
                 parent = None,
                 move: Move = False,
                 model: Model = None):

        # general tree variables
        self.parent: Node = parent
        self.children: list[Node] = []
        # node specific variables
        self.state: Chessboard = state
        self.move: Move = move
        self.v = 0
        self.p: float = p
        self.visits: int = 0
        self.value: float = 0
        # network
        self.model = model

        self.time_predicted = 0

    def ucb(self):
        return (self.p * exploration_constant * sqrt(self.parent.visits) / (1 + self.visits)
                + (self.value / self.visits if self.visits > 0 else 0))

    def __str__(self):
        """Method to return a node as a string.

        :return: String, A string representing the entire subtree from this node.
        """
        string_buffer = []
        self.print_tree(string_buffer, "", "")
        return "".join(string_buffer)
    
    def run(self, iterations:int=tree_iterations):
        """Method to run the MCTS search continually for the remaining iterations

        :return: None
        """
        for _ in range(iterations):
            self.mcts()


    def mcts(self):
        node = self.select()
        v = node.expand()
        node.backpropagate(v)

    def select(self):
        if self.children:
            node = max(self.children, key=lambda n: n.ucb())
            self.state.move(node.move)
            node.select()
        
    def expand(self):
        status = self.state.get_game_status()
        if status == 2:
            return 0.5
        elif status == 0 or status == 1:
            return 0
        else:
            p_vector, v = self.possible_moves()
            for (move, p) in p_vector:
                self.children.append(
                    Node(
                        state=self.state,
                        move=move,
                        p=p,
                        parent=self,
                        model=self.model
                    )
                )
            return v
        
    def backpropagate(self, v: float):
        """
        Method that backpropagates the value v for the current node.

        :param node: Node class, the current node to backpropagate from
        :param v: Float, the value to backpropagate up the tree, v ranges from 0 to 1 (sigmoid)
        :return: None
        """

        self.value += v
        self.visits += 1
        # if we aren't at root node, backpropagate
        if self.parent != None:
            # invert v value to because of color change before backpropagating
            self.state.unmove()
            return self.parent.backpropagate(1-v)
        else:
            return self

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

        predict_start = time.time()
        p, v = self.model.predict(input_repr, verbose=None)
        predict_end = time.time()
        self.time_predicted += (predict_end-predict_start)
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
            p = round(self.p, 10)
            val = round(self.value, 10)
            # v = round(self.v, 10)
            visits = self.visits
            if self.parent:
                if visits != 0:
                    info_text = f'(p:{p}|v:{val}|n:{visits}|wr:{val/visits}|u:{self.ucb()}|move:{self.move})'
                else:
                    info_text = f'(p:{p}|v:{val}|n:{visits}|wr:-|u:{self.ucb()}|move:{self.move})'
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

    def update_tree(self, move:Move):
        """Updates the tree based on a certain move (moves down the tree one level)

        :param new_state: New chessboard state, used only if the child doesn't exist yet, in order to create a new node
        :param move: Move that is being performed
        :return: None
        """
        # select the child that will become the new root node
        child = self.select_child(move)
        # resetting the time
        self.time_predicted = 0
        # if the child doesn't exist for this tree yet
        # for example if the opponent made a move that hasn't been explored/expanded for this player yet
        child.add_noise()
        
        self.state.move(child.move)
        child.parent = None

        return child

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
        raise RuntimeError("Child as Move: %s not found." % str(move))