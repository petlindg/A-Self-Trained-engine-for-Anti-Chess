import bz2
import multiprocessing
import pickle
import time
from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split

from config import max_buffer_size, training_iterations, games_per_iteration, checkpoint_path
from config import epochs, batch_size, verbosity
from Game.TrainingGame import TrainingGame
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE


class NeuralNetworkProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queues):
        super(NeuralNetworkProcess, self).__init__()
        self.input_queue = input_queue # shared queue for the input
        self.output_queues = output_queues # dict of queues to send back, has UID as key and queue as value
        self.list_uid = []
        self.list_states = []
        self.batch_size = 30 # number of states to process in a single batch
        self.buffer = deque(maxlen=max_buffer_size)
        self.games_counter = 0
        self.model = None
        self.evaluations = {'hits': 0, 'misses': 0}

    def run(self):
        """Start the neural network process allowing it to process data

        :return: None
        """
        start_time = time.time()
        model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
        self.model = model_config.build_nn()
        try:
            #pass
            self.model.load_weights(checkpoint_path)
        except Exception as e:
            print('EXCEPTION, couldnt load weights ', e)

        while True:
            # get the latest request from the queue
            # based on the request type, perform some action
            request_type, uid, data = self.input_queue.get()

            # if the request is an evaluation then store the data for later evaluation
            # also check if the data already exists in the dictionary, if it does,
            # return the value to the process which asked for it
            if request_type == 'eval':
                dict_vals = self.evaluations.get(data.data.tobytes())
                if dict_vals is not None:
                    self.evaluations['hits'] += 1
                    p, v = dict_vals
                    output_queue = self.output_queues[uid]
                    output_queue.put((p, v))
                # if the data doesn't exist in the dictionary, add it to the internal storage for evaluation
                # the internal data is split into two parts so that it can later be sent back to the correct worker
                else:
                    self.evaluations['misses'] += 1
                    self.list_uid.append(uid)
                    self.list_states.append(data)

            # if the request is a finished game, data will be the final result from the game
            # therefore append it to our data buffer and increment our counter
            elif request_type == 'finished':
                self.buffer.append(data)
                self.games_counter += 1
                print(f'game finished uid:{uid} | c: {self.games_counter}')

            # if the list of pending evaluation states is large enough
            # perform the model evaluation on the list
            if len(self.list_states) >= self.batch_size:
                self._process_requests()

            # if the accumulated game results reaches appropriate size, train the model
            if self.games_counter >= games_per_iteration:
                self._train_network()
                print(f'{time.time() - start_time} s')
                hits = self.evaluations['hits']
                misses= self.evaluations['misses']
                sum = hits+misses
                print(f'hits: {hits} | {misses} | hitrate: {hits/sum} | total: {sum}')
                time.sleep(30)
                # 30 games in 83 seconds, 2.7s per game

                # 120 games in 185s, 1.5s
                # 85% hitrate, 108356 state evaluations
                # 92516 hits, 15840 misses
    def _process_requests(self):
        """Method to process the requested evaluations and return the results to all the linked processes.

        :return: None
        """
        # reshape the array to fit the model
        model_input = np.array(self.list_states).reshape((-1, 8, 8, 17))
        result = self.model.predict(model_input, verbose=0)
        list_ps = result[0]
        list_vs = result[1]
        result_zip = zip(list_ps, list_vs)

        # go through the result list and send each individual result back to the corresponding process
        for input_repr, key, (p, v) in zip(model_input, self.list_uid, result_zip):
            self.evaluations[input_repr.data.tobytes()] = (p, v)
            out_queue = self.output_queues[key]
            out_queue.put((p, v))

        # empty the now processed input lists
        self.list_uid = []
        self.list_states = []

    def _train_network(self):
        """Method to train the network from the accumulated games

        :return: None
        """
        pass


class GameProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, initial_state, uid):
        super(GameProcess, self).__init__()
        self.outgoing_queue = input_queue
        self.incoming_queue = output_queue
        self.initial_state = initial_state
        self.uid = uid

    def run(self):
        # while the process is running, keep running training games
        while True:
            game = TrainingGame(initial_state=self.initial_state, outgoing_queue=self.outgoing_queue,
                                incoming_queue=self.incoming_queue, uid=self.uid)
            result = game.run()
            print(result)
            self.outgoing_queue.put(('finished', self.uid, game.get_history()))
