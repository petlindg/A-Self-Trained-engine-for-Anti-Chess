import bz2
import multiprocessing
import pickle
import time
from collections import deque

import numpy as np
from sklearn.model_selection import train_test_split

import config
from Game.state_generator import generate_random_state
from chess import Chessboard
from config import max_buffer_size, training_iterations, games_per_iteration, checkpoint_path
from config import epochs, batch_size, verbosity, train_split
from Game.TrainingGame import TrainingGame
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE

class TrainingData:
    def __init__(self):
        self.X_train = deque(maxlen=int(train_split*max_buffer_size))
        self.X_test = deque(maxlen=int((1-train_split)*max_buffer_size))
        self.y_train = []
        self.y_test = []

    def add(self, X_train, X_test, y_train, y_test):
        self.X_train += X_train
        self.X_test += X_test
        self.y_train += y_train
        self.y_test += y_test

class NeuralNetworkProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queues):
        super(NeuralNetworkProcess, self).__init__()
        self.input_queue = input_queue # shared queue for the input
        self.output_queues = output_queues # dict of queues to send back, has UID as key and queue as value
        self.list_uid = []
        self.list_states = []
        self.batch_size = len(output_queues) # number of states to process in a single batch

        self.buffer = []
        self.training_data = TrainingData()
        self.games_counter = 0
        self.model = None
        self.evaluations = {'hits': 0, 'misses': 0}
        self.eval_result = []
    def run(self):
        """Start the neural network process allowing it to process data

        :return: None
        """
        # if the past game data should be loaded or not
        self._load_past_data()
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
                self.games_counter = 0
                self._split_data()
                self._train_network()
                print(f'{time.time() - start_time} s')
                hits = self.evaluations['hits']
                misses = self.evaluations['misses']
                sum = hits+misses
                print(f'hits: {hits} | {misses} | hitrate: {hits/sum} | total: {sum}')
                self.evaluations.clear()
                self.evaluations['hits'] = hits
                self.evaluations['misses'] = misses

                # 30 games in 83 seconds, 2.7s per game

                # 120 games in 185s, 1.5s
                # 85% hitrate, 108356 state evaluations
                # 92516 hits, 15840 misses

                # ===========================================================================
                # self play training 3R1K vs 3r1k completely fresh model
                # ===========================================================================
                # loop 1 == 31s per game ==
                # 60 games took 1850s, 236k states, 73% hitrate
                # eval1_loss [3.573718547821045, 3.416172504425049, 0.15754616260528564]

                # loop 2 == 41s per game ==
                # 120 games took 4316s, 657k states, 82% hitrate
                # eval2_loss [2.009107828140259, 1.909683346748352, 0.09942435473203659]

                # loop 3 == 24s per game ==
                # 180 games took 5775s, 855k states, 81% hitrate
                # eval3_loss [1.71271550655365, 1.614129900932312, 0.09858565032482147]

                # loop 4 == 25s per game ==
                # 240 games took 7266s, 1007k states, 79% hitrate
                # eval4_loss [1.567151665687561, 1.487604022026062, 0.07954749464988708]

                # loop 5 == 29s per game ==
                # 300 games took 9000s, 1187k states, 77% hitrate
                # eval5_loss [1.5922825336456299, 1.5150401592254639, 0.07724227011203766]

                # loop 6 == ==
                # 360 games took 9000+4500s, 90% hitrate
                # eval6_loss [1.5357917547225952, 1.4594144821166992, 0.07637675106525421]

                # loop 7 == ==
                # 420 games took 9000+7800, 90% hitrate
                # eval7_loss [1.4593110084533691, 1.3918009996414185, 0.06750993430614471]

                # loop 7
                # 480 games took 9000+11000s, 82% hitrate
                # eval8_loss [1.4593110084533691, 1.3918009996414185, 0.06750993430614471]


    def _load_past_data(self):
        data = TrainingData()
        try:
            with bz2.BZ2File('Game/training_data_class.bz2', 'r') as f:
                data = pickle.load(f)
        except:
            print('couldnt load past data')
        self.training_data = data

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

    def _split_data(self):
        list_states = []
        list_outputs = []
        for game in self.buffer:
            for (state, dist, v) in game:
                list_states.append(state[0])
                list_outputs.append((np.array(dist).flatten(), v))
        X_train, X_test, y_train, y_test = train_test_split(list_states, list_outputs,
                                                            shuffle=True,
                                                            train_size=train_split)
        self.training_data.add(X_train, X_test, y_train, y_test)
        self.buffer.clear()

    def _train_network(self):
        """Method to train the network from the accumulated games

        :return: None
        """

        dists_train, vs_train = zip(*self.training_data.y_train)
        dists_test, vs_test = zip(*self.training_data.y_test)

        self.model.fit(np.array(self.training_data.X_train),
                       [np.array(dists_train), np.array(vs_train)],
                       epochs=epochs,
                       verbose=1,
                       batch_size=batch_size
                       )

        eval = self.model.evaluate(np.array(self.training_data.X_test),
                            [np.array(dists_test), np.array(vs_test)]
                            )
        print(eval)
        self.eval_result.append(eval)

        self.model.save_weights(checkpoint_path)
        save_to_file('Game/training_data_class.bz2', self.training_data)

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
            random_state = generate_random_state(config.piece_list)
            if config.random_state_generation:
                game = TrainingGame(initial_state=Chessboard(random_state), outgoing_queue=self.outgoing_queue,
                                    incoming_queue=self.incoming_queue, uid=self.uid)
            else:
                game = TrainingGame(initial_state=self.initial_state, outgoing_queue=self.outgoing_queue,
                                    incoming_queue=self.incoming_queue, uid=self.uid)
            result = game.run()
            print(result)
            self.outgoing_queue.put(('finished', self.uid, game.get_history()))
def save_to_file(filename, data):
    with bz2.BZ2File(filename, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)