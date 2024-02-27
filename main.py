import os
import time
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow

import chess
from MCTS import MCTS
from collections import deque
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
import copy
# a class representing the generation of the Training Data

checkpoint_path = "checkpoints/checkpoint.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

class TrainingData:
    def __init__(self, max_buffer_size, initial_state, tree_iterations, training_iterations, games):
        # a buffer with a max size operating on a first in first out principle
        # during the training process, the oldest data will be replaced with the newest data
        # inside this buffer
        self.buffer = deque(maxlen=max_buffer_size)
        self.initial_state = initial_state
        self.tree_iterations = tree_iterations
        self.training_iterations = training_iterations
        self.games_per_iteration = games
        self.evaluation_result = []
    # method that will continually generate training data through self play while self.training is true
    def train(self):
        # main training loop, creates a game, runs the game and saves the game data - repeat
        training_count = 0
        initial_game = Game(self.initial_state, self.tree_iterations)
        while training_count < self.training_iterations:
            game_counter = 0
            while game_counter < self.games_per_iteration:
                game = copy.deepcopy(initial_game)
                game_start_time = time.time()
                state_result = game.check_end_state()
                tree = None
                while type(state_result) == bool:
                    tree = game.make_move(tree)
                    state_result = game.check_end_state()
                    #print(game.current_state)

                game_end_time = time.time()
                # total time spent for the entire single game
                total_game_time = game_end_time - game_start_time
                print(total_game_time, 's | ', game.time_prediction, 's | thinking %: ', game.time_prediction/total_game_time)
                self.buffer.append(state_result)
                game_counter += 1
            training_count += 1

            # after a training iteration is complete, enough play data has been generated to fit the model once
            self.fit_data()
            model.save_weights(checkpoint_path)

    def test(self):
        initial_game = Game(self.initial_state, self.tree_iterations)
        game = copy.deepcopy(initial_game)
        game_start_time = time.time()
        state_result = game.check_end_state()
        tree = None
        while type(state_result) == bool:
            tree = game.make_move(tree)
            state_result = game.check_end_state()
            print(game.current_state)

        game_end_time = time.time()
        # total time spent for the entire single game
        total_game_time = game_end_time - game_start_time
        print(total_game_time, 's | ', game.time_prediction, 's | thinking %: ', game.time_prediction / total_game_time)


    def fit_data(self):
        """Method that uses  the data stored in the buffer to fit the model
           and also evaluates the model, using a train test split.

        :return:
        """
        list_states = []
        list_outputs = []
        # flattening out the buffer of games into the input and output data lists
        for game in self.buffer:
            for (state, dist, v) in game:
                list_states.append(state[0])
                list_outputs.append((np.array(dist).flatten(), v))

        print(len(self.buffer))

        X_train, X_test, y_train, y_test = train_test_split(list_states, list_outputs, shuffle=True)
        # transforming the now shuffled list of tuples into two separate lists
        dists_train, vs_train = zip(*y_train)
        dists_test, vs_test = zip(*y_test)
        model.fit(np.array(X_train),
                  [np.array(dists_train), np.array(vs_train)],
                  epochs=50,
                  verbose=0,
                  batch_size=16
                  )

        self.evaluation_result.append(model.evaluate(np.array(X_test),
                                        [np.array(dists_test), np.array(vs_test)]
                                        ))



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
        self.time_prediction = 0
    # method to check if the game is over
    # and if so, do the required actions to finalize the game
    def check_end_state(self):
        '''
        Checks if the game has reached a final state and returns a complete game history

        :return: a list of tuples [(state, mcts_distribution, reward)]
        '''

        status = self.current_state.get_game_status()

        if status != 3:
            game_over = True
        else:
            game_over = False

        if status == 0:
            winner = chess.Color.WHITE
            print('===============================')
            print('           white wins')
            print('===============================')
        elif status == 1:
            winner = chess.Color.BLACK
            print('===============================')
            print('           black wins')
            print('===============================')
        elif status == 2:
            winner = 'draw'
            print('===============================')
            print('             draw')
            print('===============================')
        finalized_history = []
        if game_over:
            for (state, mcts, v, player) in self.game_history:
                # if nobody won and it's a draw
                if winner == 'draw':

                    finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0))
                # if winning player
                elif winner == player:

                    finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 1))
                # if losing player
                elif winner is not player:
                    finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), -1))
            return finalized_history
        else:
            return True

    def make_move(self, old_tree=None):
        """
        Performs a move for the current game by creating an mcts tree

        :return: None
        """
        tree = MCTS(self.current_state, self.tree_iterations, model, old_tree)
        tree.run()
        self.time_prediction += tree.time_prediction
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
        return tree.select_child(best_move)


class PlayGame:
    '''Class representing a single game against the AI via commandline'''
    def __init__(self, initial_state, tree_iterations):
        self.current_state = initial_state
        self.game_over = False
        self.game_history = []
        self.tree_iterations = tree_iterations
        self.time_prediction = 0
        self.player_color = chess.Color.BLACK
        self.old_tree = None
    def check_end_state(self):
        '''
        Checks if the game has reached a final state

        :return: a list of tuples [(state, mcts_distribution, reward)]
        '''

        status = self.current_state.get_game_status()

        if status != 3:
            game_over = False
        else:
            game_over = True

        if status == 0:
            winner = chess.Color.WHITE
            print('===============================')
            print('           white wins')
            print('===============================')
        elif status == 1:
            winner = chess.Color.BLACK
            print('===============================')
            print('           black wins')
            print('===============================')
        elif status == 2:
            winner = 'draw'
            print('===============================')
            print('             draw')
            print('===============================')
        return game_over
    def player_move(self):
        """Takes player input and performs the players move when it is the players turn

        :param self:
        :return: None
        """
        hasnt_moved = True
        while hasnt_moved:
            possible_moves = self.current_state.get_moves()
            m_string = '| '
            for m in possible_moves:
                m_string = m_string + str(m) + ' | '
            print(m_string)
            alg_not = input('algebraic notation:')

            move = algebraic_to_bitboard(alg_not)

            for m in possible_moves:
                s = m.src_index
                d = m.dst_index
                if move.src_index == s and d == move.dst_index:
                    self.current_state.move(move)
                    hasnt_moved = False
                    if self.old_tree is not None:
                        return self.old_tree.select_child(move)

            if hasnt_moved:
                print('invalid move, try again')
    def ai_move(self, old_tree=None):
        """
                Performs a move for the current game by creating an mcts tree

                :return: None
                """
        tree = MCTS(self.current_state, self.tree_iterations, model, old_tree)
        tree.run()
        self.time_prediction += tree.time_prediction
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
        mcts_dist = [(n / visit_total, move) for (n, move) in mcts_dist]

        # add values to the game history recording all moves
        self.game_history.append((self.current_state, mcts_dist, tree.root_node.v, self.current_state.player_to_move))

        self.current_state.move(best_move)
        return tree.select_child(best_move)

    def run(self):
        """Main loop for the game against the AI, continually runs until end of game.

        :param self:
        :return: None
        """
        time.sleep(10)
        print(self.current_state)
        while self.check_end_state():


            if self.current_state.player_to_move == self.player_color:
                self.old_tree = self.player_move()
            else:
                self.old_tree = self.ai_move(self.old_tree)
            print(self.current_state)

def algebraic_to_bitboard(s):
    """Takes an algebraic representation string s and returns the move representation for this string

    :param s: String
    :return: Class Move
    """
    s = list(s.lower())
    fst_c = ''
    fst_d = 0
    snd_c = ''
    snd_d = 0
    counter = 0
    for character in s:
        if counter == 4:
            break
        if counter == 0 and character.isalpha():
            fst_c = str(character)
        if counter == 1 and character.isdigit():
            fst_d = int(character)
        if counter == 2 and character.isalpha():
            snd_c = str(character)
        if counter == 3 and character.isdigit():
            snd_d = int(character)
        counter +=1
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # default values in the case that the input is wrong and the characters arent valid
    if fst_c not in cols:
        src_col = 1
    if snd_c not in cols:
        dst_col = 1
    if fst_d > 8:
        fst_d = 0
    if snd_d > 8:
        snd_d = 0

    # convert letters to integers
    for i, c in enumerate(cols):
        if fst_c == c:
            src_col = i + 1
        if snd_c == c:
            dst_col = i + 1

    src_index = np.uint8(-src_col + 8 * fst_d)
    dst_index = np.uint8(-dst_col + 8 * snd_d)
    print(src_index, dst_index)
    return chess.Move(src_index, dst_index)

def move_to_algebraic(move):
    # d6d5 = 44, 36

    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    src = move.src_index
    src_row = str(src//8 + 1)
    src_col = cols[(src%8-1)]

    dst = move.dst_index
    dst_row = str(dst//8 + 1)
    dst_col = cols[(src%8-1)]

    return src_col + src_row + dst_col + dst_row



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
    try:
        model.load_weights(checkpoint_path)
    except:
        pass
    starting_board = chess.Chessboard()
    starting_board.init_board_test_2()
    training_session = TrainingData(20, starting_board, 180, 40, 5)
    training_session.train()

def main_play():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    global model
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except:
        pass
    starting_board = chess.Chessboard()
    starting_board.init_board_test_2()
    game = PlayGame(starting_board, 160)
    game.run()

if __name__ == "__main__":
    #main()
    main_play()