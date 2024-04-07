import bz2
import pickle
from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split

from config import max_buffer_size, training_iterations, games_per_iteration, checkpoint_path
from config import epochs, batch_size, verbosity
from Game.TrainingGame import TrainingGame


class Training:
    """Class representing the Training process of the neural network"""
    def __init__(self, initial_state, model):
        self.buffer = deque(maxlen=max_buffer_size)
        self.initial_state = initial_state
        self.model = model
        self.evaluation_result = []

    def load_from_file(self, filename):
        data = None
        try:
            with bz2.BZ2File(filename, 'r') as f:
                data = pickle.load(f)
        except Exception as e:
            print(e)
        if data is not None:
            for game in data:
                self.buffer.append(game)

    def save_to_file(self, filename, data):
        with bz2.BZ2File(filename, 'w') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

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
                result = game.run()
                if result != 'draw':
                    self.buffer.append(game.get_history())
                    game_counter += 1

            print("Training iteration: " + str(t_counter))
            list_states = []
            list_outputs = []
            for game in self.buffer:
                for (state, dist, v) in game:
                    list_states.append(state[0])
                    list_outputs.append((np.array(dist).flatten(), v))
            print(len(self.buffer))

            # split the training and testing data up, making sure to shuffle the data
            X_train, X_test, y_train, y_test = train_test_split(list_states, list_outputs, shuffle=True)

            self.fit_data(X_train, X_test, y_train, y_test)
            t_counter += 1
            self.model.save_weights(checkpoint_path)
            self.save_to_file('Game/trainingdata.bz2', list(self.buffer))

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
        list_states = []
        list_outputs = []
        # flattening out the buffer of games into the input and output data lists
        for game in data:
            for (state, dist, v) in game:
                list_states.append(state[0])
                list_outputs.append((np.array(dist).flatten(), v))
        print(len(data))

        # split the training and testing data up, making sure to shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(list_states, list_outputs, shuffle=True)

        counter = 0
        # 19 loops so far
        while counter < 200:
            if data is not None:
                self.fit_data(X_train, X_test, y_train, y_test)
                self.model.save_weights(checkpoint_path)
            counter += 1
            print(f'loop:{counter}')


    def fit_data(self, X_train, X_test, y_train, y_test):
        """Method that uses  the data stored in the buffer to fit the model
           and also evaluates the model, using a train test split.

        :return:
        """

        # transforming the now shuffled list of tuples into two separate lists
        dists_train, vs_train = zip(*y_train)
        dists_test, vs_test = zip(*y_test)
        self.model.fit(np.array(X_train),
                       [np.array(dists_train), np.array(vs_train)],
                       epochs=epochs,
                       verbose=1,
                       batch_size=batch_size
                       )

        # store the results of the evaluation
        self.evaluation_result.append(self.model.evaluate(np.array(X_test),
                                      [np.array(dists_test), np.array(vs_test)]
                                        ))


