import bz2
import pickle
from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split

from Game.Utils import translate_moves_to_output
from chess import Chessboard
from config import max_buffer_size, training_iterations, games_per_iteration, checkpoint_path
from config import epochs, batch_size, verbosity
from Game.TrainingGame import TrainingGame
from nn_architecture import OUTPUT_SHAPE, INPUT_SHAPE, NeuralNetwork


class Training:
    """Class representing the Training process of the neural network"""
    def __init__(self, initial_state, model):
        self.buffer = deque(maxlen=max_buffer_size)
        self.initial_state = initial_state
        self.model = model
        self.evaluation_result = []

    def train(self):
        """Method that performs the training in accordance with the config

        :return: None
        """
        t_counter = 0

        # Outer loop that performs training_iterations
        while t_counter < training_iterations:
            game_counter = 0
            # inner loop where each training_iteration performs a number of games
            while game_counter < games_per_iteration:
                game = TrainingGame(initial_state=self.initial_state, model=self.model)
                game.run()
                self.buffer.append(game.get_history())
                game_counter += 1
            print("Training iteration: " + str(t_counter))
            self.fit_data(self.buffer)
            t_counter += 1
            self.model.save_weights(checkpoint_path)

    def train_from_file(self, filename):
        """Method that performs model fitting based on a compressed pickle file

        :param filename: String, path to the bz2 pickle file.
        :return: None
        """
        data = None
        try:
            with bz2.BZ2File(filename, 'r') as f:
                data = pickle.load(f)
        except Exception as e:
            print(e)
        if data is not None:
            self.fit_data(data)
            self.model.save_weights(checkpoint_path)

    def test_train(self, fen1, fen2):
        """ function to train the network on two individual states (set both to the same if you want one state)

        :param fen1:
        :param fen2:
        :return:
        """
        # for first fen
        board1 = Chessboard(fen1)
        t_board1 = board1.translate_board()[0]
        moves1 = board1.get_moves()

        sum_v = sum(range(1,len(moves1)+1))
        vs = range(1,len(moves1)+1)
        movelist = [(v/sum_v, m) for m, v in zip(moves1, vs)]
        print(movelist)
        moves1 = translate_moves_to_output(movelist)

        # for second fen
        board2 = Chessboard(fen2)
        t_board2 = board2.translate_board()[0]
        moves2 = board2.get_moves()

        sum_v = sum(range(1, len(moves2) + 1))
        vs = range(1, len(moves2) + 1)
        movelist = [(v / sum_v, m) for m, v in zip(moves2, vs)]
        print(movelist)
        moves2 = translate_moves_to_output(movelist)


        inputlist = np.array([t_board1, t_board2])
        outputlist = [np.array([np.array(moves1).flatten(), np.array(moves2).flatten()]), np.array([0.2, 0.8])]
        print(inputlist)
        print(outputlist)
        self.model.fit(inputlist,
                       outputlist,
                       epochs=epochs,
                       verbose=verbosity,
                       batch_size=batch_size
                       )
        self.model.save_weights(checkpoint_path)

    def test_trained(self, fen):
        board = Chessboard(fen)
        game = TrainingGame(board, self.model)
        game.run()

    def fit_data(self, buffer):
        """Method that uses  the data stored in the buffer to fit the model
           and also evaluates the model, using a train test split.

        :return:
        """
        list_states = []
        list_outputs = []
        # flattening out the buffer of games into the input and output data lists
        for game in buffer:
            for (state, dist, v) in game:
                list_states.append(state[0])
                list_outputs.append((np.array(dist).flatten(), v))

        print(len(buffer))

        # split the training and testing data up, making sure to shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(list_states, list_outputs, shuffle=True)
        # transforming the now shuffled list of tuples into two separate lists
        dists_train, vs_train = zip(*y_train)
        dists_test, vs_test = zip(*y_test)
        self.model.fit(np.array(X_train),
                       [np.array(dists_train), np.array(vs_train)],
                       epochs=epochs,
                       verbose=verbosity,
                       batch_size=batch_size
                       )

        # store the results of the evaluation
        self.evaluation_result.append(self.model.evaluate(np.array(X_test),
                                      [np.array(dists_test), np.array(vs_test)]
                                        ))


def main():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    model.load_weights(checkpoint_path)
    fen1 = "K7/8/8/8/8/8/8/7r w - 0 1"
    fen2 = "K7/8/8/8/8/8/8/7r b - 0 1"
    t = Training(Chessboard(fen1), model)
    #t.test_train(fen1, fen2)
    t.test_trained(fen2)

"""
 using only one state to train the network:
 fen: "K7/8/8/8/8/8/8/7r w - 0 1" => loss: 1.0 | pred: dist: [0.16, 0.33, 0.5]| v: 0.8
 fen: "K7/8/8/8/8/8/8/7r b - 0 1" => loss: 2.5 | pred: dist: [0.009520664286358943, 0.01905778648903371, 0.02858673934343422, 0.03808218771760198, 0.04760269459661189, 0.05712322755494591, 0.06670249860238807, 0.07616422641049482, 0.08576041921366, 0.0952051954611019, 0.10472572096820046, 0.11434716600832968, 0.12387623063126205, 0.13324524271657637]| v: 0.8

 fen: "k7/8/8/8/8/8/8/7R w - 0 1" => loss: 2.5 | pred: dist: [0.009519158568450975, 0.019038125267562843, 0.02858283133165692, 0.038110511320416283, 0.04763825464468564, 0.05711399578962854, 0.0666329578317176, 0.07615195712998903, 0.08574897832534133, 0.09518991101911306, 0.10480430849791417, 0.11422789471318295, 0.12385976534150701, 0.13338135021883363]| v: 0.8
 fen: "k7/8/8/8/8/8/8/7R b - 0 1" => loss: 1.0 | pred: dist: [0.16, 0.33, 0.5]| v: 0.8
 conclusion: overfitting the network on a singular state regardless of color is possible and the outcome is as expected.

    
# =================================================================================
# Testing on two states instead of just one
# =================================================================================
    
    test version 1:
    fen1 = "k7/8/8/8/8/8/8/7R b - 0 1" v=0.8 (rook player has higher win value)
    fen2 = "k7/8/8/8/8/8/8/7R w - 0 1" v=0.2
    
    result:
    p loss = 2.4376
    v loss = 0.09
    
    fen1 test:
    dist = [0.1665853383020533, 0.33317034874727397, 0.5002443129506727]
    v = 0.5
    
    fen2 test:
    dist = [0.009528314030792412, 0.019056829247520963, 0.028557831356066, 0.03811385595531258, 0.04764234602379911, 0.05717089197726791, 0.0666346150769793, 0.0762279354505542, 0.08567300414318685, 0.09519218377362867, 0.10471145282004213, 0.11434195533514808, 0.12374979717826386, 0.13339898763143793]
    v = 0.5

# =================================================================================

    test version 2:
    fen1 = "K7/8/8/8/8/8/8/7r w - 0 1" v=0.2
    fen2 = "K7/8/8/8/8/8/8/7r b - 0 1" v=0.8 (rook player has higher win value)

    result:
    p loss = 2.4375
    v loss = 0.09
    
    fen1 test:
    dist = [0.16680166194608045, 0.3332791503714042, 0.4999191876825153]
    v = 0.5
    
    fen2 test:
    dist = [0.009525389347540556, 0.01905097981773042, 0.02854919060503627, 0.0380655527114423, 0.047627746572683136, 0.057153351009723644, 0.06667895917125766, 0.07620450774089559, 0.08573015314736467, 0.09525567192105454, 0.10478129498056259, 0.11430687334614856, 0.12371235169414868, 0.13335797793441143]
    v = 0.5




    """



if __name__ == '__main__':
    main()

