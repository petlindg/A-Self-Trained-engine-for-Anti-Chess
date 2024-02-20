import chess
from MCTS import MCTS
from collections import deque
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE

# a class representing the generation of the Training Data
class TrainingData:
    def __init__(self, max_buffer_size, initial_state, tree_iterations):
        # a buffer with a max size operating on a first in first out principle
        # during the training process, the oldest data will be replaced with the newest data
        # inside this buffer
        self.buffer = deque(max_buffer_size)
        self.training = True
        self.initial_state = initial_state
        self.tree_iterations = tree_iterations

    # method that will continually generate training data through self play while self.training is true
    def train(self):
        while self.training:
            pass


# class representing an entire game session
class Game:
    def __init__(self, initial_state, tree_iterations):
        self.current_state = initial_state
        self.game_over = False
        self.game_history = []
        self.tree_iterations = tree_iterations

    # method to check if the game is over
    # and if so, do the required actions to finalize the game
    def check_end_state(self):
        pass

    # predict the best move for the current player and make said move
    def make_move(self):
        tree = MCTS(self.current_state, self.tree_iterations, model)
        tree.run()
        potential_nodes = tree.root_node.children
        max_visit = 0
        best_move = None
        # iterate through all the possible children to the current state
        # and select the one with the highest visit count, this is the move to perform
        mcts_dist = []
        visit_total = 0
        for node in potential_nodes:
            visit_total += node.visits
            mcts_dist.append((node.visits, node.move))
            if node.visits > max_visit:
                max_visit = node.visits
                best_move = node.move
        # prepare the float distribution of all actions
        # so that the model can use it for backpropagation
        mcts_dist = [(n/visit_total, move) for (n, move) in mcts_dist]

        # add values to the game history recording all moves
        self.game_history.append((self.current_state, mcts_dist, tree.root_node.v, self.current_state.player_to_move))

        self.current_state.move(best_move)



def main():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    global model
    model = model_config.build_nn()
    starting_board = chess.Chessboard()
    starting_board.init_board_standard()
    game1 = Game(starting_board, 40)
    while True:
        print(game1.current_state)
        game1.make_move()

if __name__ == "__main__":
    main()