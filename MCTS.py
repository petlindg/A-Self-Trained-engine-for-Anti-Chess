import time
from math import sqrt
import random

import chess
from chess import Chessboard
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
import copy

# Function that takes a chessboard as a current state, a reference to the model
# and then calculates the new possible moves for the current state, gets the model results for
# the current state and then puts them together as a tuple and returns it.
def possible_moves(state: Chessboard, model):
    # translate the board to the input representation
    input_repr = state.translate_board()
    # get the valid moves for the current chessboard
    moves = state.get_moves()
    # get the model predictions for the current state
    time_predict_start = time.time()
    p, v = model.predict(input_repr, verbose = None)
    time_predict_end = time.time()
    time_predict = time_predict_end - time_predict_start
    v = v[0][0]
    p_array = p.reshape(1,1,8,8,76)

    # for every possible move, create a new chessboard based on that state
    # and append this state, the value p from the model and the move to the return list
    return_list = []
    for move in moves:
        new_state = copy.deepcopy(state)
        new_state.move(move)
        p_val = fetch_p_from_move(move, p_array)
        return_list.append((new_state, move, p_val))

    return ((return_list, v), time_predict)

# Function that returns the singular p value for a given move
# from the model output.
def fetch_p_from_move(move: chess.Move, model_output):
    """Function that returns the P value for a specific move
    from the model output.

    :param move: Move class, The move to fetch the value from
    :param model_output: array of shape (1x1x8x8x76), The output from the model from which to fetch the P value
    :return: Float value P, from the model_output
    """
    src_col = move.src_index % 8
    src_row = move.src_index // 8

    move_type = chess.calc_move(move.src_index, move.dst_index, move.promotion_type)
    return model_output[0][0][src_row][src_col][move_type]

def ucb(node, c):
    """Calculates the upper confidence bound for trees

    :param node: Node class, The current node to calculate the UCB for.
    :param c: Float, the exploration constant
    :return: Float, the UCB score
    """
    return (node.p * c * sqrt(node.parent.visits)/(1+node.visits) + (node.value/node.visits if node.visits > 0 else 0))

# class defining the contents of a singular node
class Node:
    """
    Class representing a single node in the tree. Includes references to its parent and children
    and the values for the node itself, such as what move was made to get to this node, the current node's state
    the value of the current node, how many visits etc.
    """
    def __init__(self, p, state, parent_node, player, root_node, move):
        self.root_node = root_node
        self.parent = parent_node
        self.state = state
        self.move = move # move that was made to get to this node
        self.children = []
        self.value = 0
        self.visits = 0
        self.v = 0
        self.p = p
        self.leaf = True
        # player is either white or black
        # TODO figure out how this will interact with the model when it comes to evaluations
        # how the player that is currently trying to find the best move will interact with the model
        self.player = player

    # string method in order to represent the tree from the current node and down
    # when you try to print a particular node
    def __str__(self):
        """Method to return a node as a string.

        :return: String, A string representing the entire subtree from this node.
        """
        string_buffer = []
        self.print_tree(string_buffer, "", "")
        return "".join(string_buffer)

    # method that can be called on a node which will print it with a given depth
    # any nodes that exist deeper than this depth will not be printed
    def print_selectively(self, depth):
        """Method to print the subtree from the node with a given depth from the current node.

        :param depth: Integer, parameter to define how deeply to print
        :return: String, string representing the limited subtree from this node
        """
        string_buffer = []
        self.print_tree(string_buffer, "", "", depth)
        print("".join(string_buffer))

    # method that will iteratively go through the tree and add on objects to the string_buffer
    # this buffer will then be used to print the entire tree in the terminal
    # depth parameter determines how deep the tree will print from the node, if depth=None then
    # all children will be printed
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
            v = round(self.value, 10)
            V = round(self.v, 10)
            try:
                U = round(ucb(self, sqrt(2)), 2)
            except AttributeError:
                U = "-"
            visits = self.visits

            info_text = f'(p:{p}|v:{v}|n:{visits}|V:{V}|U:{U})'
            string_buffer.append(info_text)
            string_buffer.append('\n')

            for i in range(0, len(self.children)):
                if i == len(self.children)-1:
                    self.children[i].print_tree(string_buffer, child_prefix + "└── ", child_prefix + "    ", (depth - 1 if depth else None))
                else:
                    self.children[i].print_tree(string_buffer, child_prefix + "├── ", child_prefix + "│   ", (depth - 1 if depth else None))


    # expand method which does both the expansion and the backpropagation, using backpropagate
    def expand(self, model):
        """Method that attempts to expand the current node in accordance to mcts using the neural network

        :param model: A reference to the neural network object.
        :return: None
        """
        self.leaf = False
        # TODO implement possible_moves
        (new_states, v), time_predict = possible_moves(self.state, model)

        # the initial value v from the model is stored separate from the value variable
        # so that it can be used in backpropagation when training the actual model.
        self.v = v
        # creating the new child nodes, making sure to swap around the player variable
        if self.player == chess.Color.WHITE:
            new_player = chess.Color.BLACK
        else:
            new_player = chess.Color.WHITE

        for (new_state, move, p) in new_states:
            self.children.append(Node(p, new_state, self, new_player, self.root_node, move))

        # backpropagation process
        self.backpropagate(self, v)
        return time_predict
    # backpropagation function, node is the current node to backpropagate to
    # v is the result value from the model and player is the player that performed the move
    # that resulted in the v value.
    def backpropagate(self, node, v):
        """Method that backpropagates the value v for the current node.

        :param node: Node class, the current node to backpropagate from
        :param v: Float, the value to backpropagate up the tree
        :param player: Player, the player that performed the move, determines how the value is backpropagated
                       for the current node
        :return: None
        """
        # if the actor performing the action with result v is the same as the root node
        # then we increase the value for the node by v
        if node.player == node.root_node.player:
            node.value += v
        # if the actor performing the action with result v is the opposite of the root node
        # then increase the value by (1-v) to get the oppositions value
        else:
            node.value += (1-v)

        node.visits += 1
        # if the parent is not none we aren't at the root yet and should continue
        if node.parent is not None:
            self.backpropagate(node.parent, v)


# class defining a singular MCTS search
class MCTS:
    """
    Class representing a single MCTS tree.
    Contains methods to perform search, selection and expansion and
    values representing the root node, exploration constant etc.
    """
    def __init__(self, root_state, iterations, model):
        # defining the root node of the tree
        self.root_node = Node(1, root_state, None, root_state.get_player(), None, None)
        self.root_node.root_node = self.root_node
        # number of MCTS iterations left, each iteration is one search
        self.iterations = iterations
        self.exploration_constant = sqrt(2)
        self.model = model
        self.time_prediction = 0
    # method to perform a single search 'iteration'
    # goes through the 3 steps of an AlphaZero MCTS loop
    # selection, expansion, backpropagation,
    # backpropagation occurs inside the expand() function
    def search(self):
        """Method that performs a single search and then tries to expand the found leaf node

        :return: None
        """
        leaf_node, w = self.selection(self.root_node)
        if w == 0:
            self.time_prediction += leaf_node.expand(self.model)
        elif w == 1:
            status = leaf_node.state.get_game_status()
            if status == 0:
                winner = chess.Color.WHITE
            elif status == 1:
                winner = chess.Color.BLACK
            else:
                winner = 'draw'

            if winner == 'draw':
                leaf_node.backpropagate(leaf_node, 0)
            # if the winner is the same as the root node, backpropagate a 100% win value from the leaf node
            elif winner == self.root_node.player:
                leaf_node.backpropagate(leaf_node, 1)
            elif winner != self.root_node.player:
                leaf_node.backpropagate(leaf_node, -1)



    # TODO look into how end of game states are handled with alphazero MCTS
    # method that performs the selection process
    # goes down the tree based off of UCB until it hits a leaf node.
    def selection(self, current_node):
        """Method that moves through the current node until it finds a leaf node

        :param current_node: Node class, the node to run the selection search from.
        :return: Tuple (Node, Integer), a tuple representing the leaf node and an integer representing
                 whether the leaf node has any children.
        """
        while not current_node.leaf:
            if len(current_node.children) != 0:
                current_node = max(current_node.children, key=lambda node: ucb(node, self.exploration_constant))
            else:
                return (current_node, 1)
        return (current_node, 0)

    # run method that will continually perform tree searches
    # until the iterations runs out
    def run(self):
        """ Runs the MCTS search for the total number of iterations as defined by the iterations value

        :return: None
        """
        for i in range(0, self.iterations):
            self.search()

def main():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    root_state = Chessboard()
    root_state.init_board_standard()
    tree = MCTS(root_state, 800, model)
    start = time.time()
    tree.run()
    end = time.time()
    print(end-start)
    # tested it with 800 iterations and it took 50 seconds
    # 16 iterations per second
    tree.root_node.print_selectively(3)


# performance metrics: 100000 iterations takes about 3.6s on my (liam's) pc
# comparing that to my java implementation, which takes 0.272s on my pc
# the java implementation is roughly 13 times faster than the python one

if __name__ == "__main__":
    main()