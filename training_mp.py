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
        self.max_size = 30
        self.buffer = deque(maxlen=max_buffer_size)
        self.games_counter = 0
        self.model = None

    def run(self):
        print('running nn process')
        start_time = time.time()
        model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
        self.model = model_config.build_nn()
        try:
            #pass
            self.model.load_weights(checkpoint_path)
        except Exception as e:
            print('EXCEPTION, couldnt load weights ', e)
        # accept new requests and store them
        # once the stored values exceed a certain size
        # process the requests
        # once finished, return all the processes correctly
        while True:
            # request_type is whether it is an evaluation or a finished game
            request_type, uid, data = self.input_queue.get()
            #print(f'request recieved: {request_type} | {uid} | {np.shape(data)}')
            # if the request is an evaluation then data will be the state to be evaluated.
            if request_type == 'eval':
                self.list_uid.append(uid)
                self.list_states.append(data)

            # if the request is a finished game, data will be the final result from the game
            elif request_type == 'finished':
                self.buffer.append(data)
                self.games_counter += 1
                print(f'game finished uid:{uid} | c: {self.games_counter}')

            # if the list states is large enough
            # perform the calculation on the set
            if len(self.list_states) >= self.max_size:
                self._process_requests()

            # if the accumulated game results reaches appropriate size, train the network
            if self.games_counter >= games_per_iteration:
                self._train_network()
                print(f'30 games done, {time.time() - start_time} s')
                # 30 games in 83 seconds, 2.7s per game

    def _process_requests(self):
        """Method to process the requested evaluations and return the results to all the linked processes.

        :return: None
        """
        model_input = np.array(self.list_states).reshape((-1, 8, 8, 17))

        result = self.model.predict(model_input, verbose=0)
        list_ps = result[0]
        list_vs = result[1]
        result_zip = zip(list_ps, list_vs)


        # go through the result and send it back to the waiting threads
        for key, (p, v) in zip(self.list_uid, result_zip):
            out_queue = self.output_queues[key]
            #print(np.shape(result_eval[0]), np.shape(result_eval[1]))
            out_queue.put((p, v))

        # empty the now processed input list
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
        print(f'running worker process {self.uid}')
        while True:
            game = TrainingGame(initial_state=self.initial_state, outgoing_queue=self.outgoing_queue,
                                incoming_queue=self.incoming_queue, uid=self.uid)
            result = game.run()
            print(result)
            self.outgoing_queue.put(('finished', self.uid, game.get_history()))
