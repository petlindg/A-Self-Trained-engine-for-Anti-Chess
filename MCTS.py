import copy
import time
from math import sqrt
import numpy as np
import chess
from Node import Node
from config import tree_iterations, exploration_constant, output_representation
from chess import Chessboard, Move, Color
from keras.models import Model

from nn_architecture import NeuralNetwork, OUTPUT_SHAPE, INPUT_SHAPE


def fetch_p_from_move(move: Move, model_output: np.array):

    src_col = move.src_index % 8
    src_row = move.src_index // 8

    move_type = chess.calc_move(move.src_index, move.dst_index, move.promotion_type)
    return model_output[0][0][src_row][src_col][move_type]


def ucb(node: Node, inverted: bool):
    if not inverted:
        return (node.p * exploration_constant * sqrt(node.parent.visits) / (1 + node.visits)
                + (node.value / node.visits if node.visits > 0 else 0))

    if inverted:
        return (node.p * exploration_constant * sqrt(node.parent.visits) / (1 + node.visits)
                + ((1 - node.value) / node.visits if node.visits > 0 else 0))


class MCTS:
    def __init__(self, root_state: Chessboard,
                 player: Color, # the player that the tree will try to maximize wins for
                 model: Model,
                 swap: bool,
                 old_tree_root: Node = None):
        # if the tree is entirely new with no old tree to reuse
        if old_tree_root is None:
            self.root_node = Node(
                p=1,
                parent=None,
                player=player,
                root_node=None,
                state=root_state
            )
            self.remaining_iterations = tree_iterations
        # if the tree is reusing old data
        else:
            self.root_node = old_tree_root
            # add noise to the new root node
            self.root_node.add_noise()
            self.remaining_iterations = tree_iterations - self.root_node.visits
        # the player that the tree is attempting to fetch the best move for
        self.tree_player = player
        self.time_predicted = 0
        self.model = model
        # a boolean indicating whether to reverse the selection or not, important variable
        # for keeping the old tree data, to keep it, we need to keep track of who originally created
        # the data and to keep track of what values belong to what player
        self.swap = swap


    def __str__(self):
        pass

    def search(self):

        # if the tree's color matches the player of the original root node we perform the search as normal
        # using the previous v values if they exist
        if self.swap:
            leaf_node, game_over = self.selection(False)

        # if the tree's color is not the same as  the original root node, we are reusing an old tree
        # but from the other player's perspective, we therefore need to invert the v values during the search phase
        else:
            leaf_node, game_over = self.selection(True)

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
                    # player doesn't matter in the backpropagation
                    leaf_node.end_state = 'draw'
                    leaf_node.backpropagate(leaf_node, 0.5, self.tree_player)
                    return

    def possible_moves(self, state: Chessboard):

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

    def selection(self, inverted: bool):
        current_node = self.root_node

        while not current_node.leaf:
            # if the selection algorithm encounters a terminal node (an end state)
            # return the node and endstate boolean set to true
            if current_node.terminal:
                return current_node, True
            else:
                current_node = max(current_node.children,
                                   key=lambda node: ucb(node, inverted))
        return current_node, False

    def run(self):
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
