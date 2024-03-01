import copy
from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split

from config import max_buffer_size, training_iterations, games_per_iteration, checkpoint_path
from config import epochs, batch_size, verbosity
from game import TrainingGame


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
        initial_game = TrainingGame(self.initial_state)

        # Outer loop that performs training_iterations
        while t_counter < training_iterations:
            game_counter = 0
            # inner loop where each training_iteration performs a number of games
            while game_counter < games_per_iteration:
                game = copy.deepcopy(initial_game)
                results = game.run(model=self.model)
                self.buffer.append(results)
                game_counter += 1
            self.fit_data()
            t_counter += 1

            self.model.save_weights(checkpoint_path)

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






