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
        # main training loop, creates a game, runs the game and saves the game data - repeat
        while self.training:
            game = Game(self.initial_state, self.tree_iterations)
            while game.check_end_state():
                game.make_move()
            # TODO check if we need to perform a copy function
            self.buffer.append(game.game_history)


# class representing an entire game session
class Game:
    """
    Class representing a single game session, contains the current state, the game history and methods
    for checking if the game has ended and making new moves for the current state.
    """
    def __init__(self, initial_state, tree_iterations):
        self.current_state = initial_state
        self.game_over = False
        self.game_history = []
        self.tree_iterations = tree_iterations

    # method to check if the game is over
    # and if so, do the required actions to finalize the game
    def check_end_state(self):
        '''
        Checks if the game has reached a final state and returns a complete game history

        :return: a list of tuples [(state, mcts_distribution, reward)]
        '''
        game_over, winner = False, 'white'
        finalized_history = []
        if game_over:
            for (state, mcts, v, player) in self.game_history:
                # if nobody won and it's a draw
                if winner == 'draw':
                    finalized_history.append((state, mcts, 0))
                # if winning player
                if winner == player:
                    finalized_history.append((state, mcts, 1))
                # if losing player
                else:
                    finalized_history.append((state, mcts, -1))
        return finalized_history


    def make_move(self):
        """
        Performs a move for the current game by creating an mcts tree

        :return: None
        """
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


# translates the bitboard move representation into the output representation for the neural network
# returns the output as an array of shape (1,1,8,8,73)
def translate_moves_to_output(mcts_dist):
    """Translates a list of moves into the output representation for the neural network

    :param mcts_dist: list of tuples [(value, move)], value is the visit % for the corresponding move.
    :return: array of shape (1x1x8x8x76), the shape of the output from the neural network.
    """
    output = [[
        [[[0 for i in range(76)] for i in range(8)] for i in range(8)]
    ]]
    # fetch all the available moves
    # for every move, calculate what type value it has and set
    # the array index as 1 for the given move
    for (val, move) in mcts_dist:
        src_col = move.src_index % 8
        src_row = move.src_index // 8
        type_value = chess.calc_move(move.src_index, move.dst_index, move.promotion_type)
        output[0][0][src_row][src_col][type_value] = val

    # return all the moves in output representation
    return output

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