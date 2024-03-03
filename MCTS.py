import copy
import time
from math import sqrt
import numpy as np
import chess
from node import Node
from config import tree_iterations, exploration_constant, output_representation
from chess import Chessboard, Move, Color
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


def ucb(node: Node):
    return (node.p * exploration_constant * sqrt(node.parent.visits) / (1 + node.visits)
            + (node.value / node.visits if node.visits > 0 else 0))


class MCTS:
    """Class representing a single MCTS tree"""
    def __init__(self, root_state: Chessboard,
                 player: Color,  # the player that the tree will try to maximize wins for
                 model: Model = None,
                 ):

        # creating the empty root node
        self.root_node = Node(
                p=1,
                parent=None,
                player=player,
                root_node=None,
                state=root_state
            )
        self.remaining_iterations = tree_iterations
        self.tree_player = player
        self.time_predicted = 0
        self.model = model

    def update_tree(self, move: Move, new_state: Chessboard):
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
        if child is None:
            new_node = Node(
                p=1,
                parent=None,
                player=self.root_node.player,
                root_node=None,
                state=new_state
            )
            self.root_node = new_node
            self.root_node.add_noise()
            self.remaining_iterations = tree_iterations
        # if the child exists in the current tree, change the root node to be this child
        else:
            self.root_node = child
            # remove the parent of the root node
            self.root_node.parent = None
            # add noise to the new root node
            self.root_node.add_noise()
            # update the iterations counter to account for the visits of the new root node
            # (if the new root node has been visited a lot already, we won't need to run the tree search on this
            # root node as much as we otherwise would)
            self.remaining_iterations = tree_iterations - self.root_node.visits

    def __str__(self):
        pass

    def search(self):
        """Search method, will find the leaf_node and check if the leaf is a terminal node
           if the leaf node is terminal then backpropagate the appropriate values. If it isn't
           terminal then the leaf node is expanded.

        :return: None
        """
        leaf_node, game_over = self.selection()

        # if there are no possible states from the selected leaf node
        # then set the node to terminal and fetch the status for the leaf node
        # and backpropagate accordingly

        # if the selection algorithm hasn't returned an end state, expand the leaf node
        if not game_over:
            new_states, v = self.possible_moves(leaf_node.state)
            leaf_node.expand(new_states, v)

        # if an end state was encountered
        elif game_over:
            # if the end state has been encountered before
            if leaf_node.end_state is not None:
                # backpropagate 0.5
                if leaf_node.end_state == 'draw':
                    leaf_node.backpropagate(leaf_node, 0.5, self.tree_player)
                # backpropagate 1 to the winning player
                else:
                    leaf_node.backpropagate(leaf_node, 1, leaf_node.end_state)

            # if this end state is new
            else:
                status = leaf_node.state.get_game_status()
                # white winning
                if status == 0:
                    leaf_node.end_state = Color.WHITE
                    leaf_node.backpropagate(leaf_node, 1, Color.WHITE)

                # black winning
                elif status == 1:
                    leaf_node.end_state = Color.BLACK
                    leaf_node.backpropagate(leaf_node, 1, Color.BLACK)
                # draw
                else:
                    # (player technically doesn't matter in the backpropagation)
                    leaf_node.end_state = 'draw'
                    leaf_node.backpropagate(leaf_node, 0.5, self.tree_player)
                    return

    def possible_moves(self, state: Chessboard):
        """Calculates all possible moves for a given chessboard using the neural network, and returns
           it as a list of tuples.

        :param state: Chessboard, the input state to calculate moves from
        :return: (list[Chessboard, Move, float], float), returns a list of new chessboards, the move
                 that was taken to get there and the float value P for that new state. In addition to this,
                 it also returns another float which is the value V from the neural network for the input state.
        """
        input_repr = state.translate_board()
        moves = state.get_moves()

        predict_start = time.time()
        p, v = self.model.predict(input_repr, verbose=None)
        predict_end = time.time()
        self.time_predicted += (predict_end-predict_start)
        v = v[0][0]
        p_array = p.reshape(output_representation)
        return_list = []

        p_sum = 0
        for move in moves:
            new_state = copy.deepcopy(state)
            new_state.move(move)
            p_val = fetch_p_from_move(move, p_array)
            p_sum += p_val
            return_list.append((new_state, move, p_val))

        # normalize the P values in the return list
        return_list = [(new_state, move, p_val/p_sum) for (new_state, move, p_val) in return_list]

        return return_list, v

    def selection(self):
        """Method that is called from the search() method, traverses the tree until it hits a leaf node

        :return: (Node, bool), returns the leaf node that was found and whether that node is terminal or not.
        """
        current_node = self.root_node

        while not current_node.leaf:
            # if the selection algorithm encounters a terminal node (an end state)
            # return the node and endstate boolean set to true
            if current_node.terminal:
                return current_node, True
            else:
                current_node = max(current_node.children,
                                   key=lambda node: ucb(node))
        return current_node, False

    def run(self):
        """Method to run the MCTS search continually for the remaining iterations

        :return: None
        """
        for i in range(0, self.remaining_iterations):
            self.search()

    def select_child(self, move):
        for child in self.root_node.children:
            if child.move == move:
                return child


def main():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    root_state = Chessboard()
    root_state.init_board_standard()
    tree = MCTS(model=model,
                player=Color.WHITE,
                root_state=root_state
                )
    tree.run()
    tree.root_node.print_selectively(4)


if __name__ == '__main__':
    main()
