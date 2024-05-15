import bz2
import multiprocessing
import os
import pickle
import random
import time
from collections import deque

import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split

from config import max_buffer_size, games_per_iteration, checkpoint_path
from config import epochs, batch_size, train_split, nn_batch


from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE


class TrainingData:
    """
    Class representing the training data, it contains all of the necessary data to train and test the neural network 
    """
    def __init__(self):
        train_size = int(train_split*max_buffer_size)
        test_size = int((1-train_split)*max_buffer_size)
        print(train_size, test_size)
        self.X_train = deque(maxlen= train_size)
        self.X_test = deque(maxlen= test_size)
        self.y_train = deque(maxlen= train_size)
        self.y_test = deque(maxlen= test_size)

    def add(self, X_train, X_test, y_train, y_test):
        self.X_train += X_train
        self.X_test += X_test
        self.y_train += y_train
        self.y_test += y_test

class NeuralNetworkProcess(multiprocessing.Process):
    """
    Class that represents a single neural network process, it handles incoming requests from
    the game processes and manages the training and execution of the neural network.
    """
    def __init__(self, input_queue, output_queues, model=None):
        super(NeuralNetworkProcess, self).__init__()
        self.input_queue = input_queue # shared queue for the input
        self.output_queues = output_queues # dict of queues to send back, has UID as key and queue as value
        self.list_uid = []
        self.list_states = []
        self.batch_size = nn_batch # number of states to process in a single batch

        self.buffer = []
        self.training_data = TrainingData()
        self.games_counter = 0
        self.model = model
        self.evaluations = {'hits': 0, 'misses': 0}
        self.eval_result = []
        self.eval_time = 0
        self.start_time = None
        self.total_iterations = 140
        self._get_old_iter()
        

    def run(self):
        """Start the neural network process allowing it to process data

        :return: None
        """

        # if the past game data should be loaded or not
        random.seed()
        self._load_past_data()
        self.start_time = time.time()

        # if there is an .h5 file in saved_model, convert it to weights
        if os.path.isfile('saved_model/model_140_it.h5'):
            h5_to_weights()
        else:
            raise RuntimeError("Checkpoints not found")

        model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
        self.model = model_config.build_nn()
        try:
            self.model.load_weights(checkpoint_path)
        except Exception as e:
            raise RuntimeError('EXCEPTION, couldnt load weights ', e)



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
                game_data, result = data
                self.buffer.append(game_data)
                self.games_counter += 1
                print(f'{result}| game finished uid:{uid} | c: {self.games_counter}')

            # if the list of pending evaluation states is large enough
            # perform the model evaluation on the list
            if len(self.list_states) >= self.batch_size:
                self._process_requests()
                end_time = time.time()
                #print(f'{self.eval_time/(end_time-self.start_time)} % spent predicting')

            # if the accumulated game results reaches appropriate size, train the model
            if self.games_counter >= games_per_iteration:
                
                hits = self.evaluations['hits']
                misses = self.evaluations['misses']
                evals = misses + hits
                tdelta = time.time()-self.start_time
                print(f'{self.games_counter} played, {evals} evaluations in {tdelta}, {evals/tdelta} /s')
                self.games_counter = 0
                self._split_data()
                self._train_network()
                print(f'{time.time() - self.start_time} s')
                
                print(f'hits: {hits} | {misses} | hitrate: {hits/evals}')
                self.evaluations.clear()
                self.evaluations['hits'] = hits
                self.evaluations['misses'] = misses

    def _get_old_iter(self):
        try:
            with bz2.BZ2File('Game/iterations_counter.bz2', 'r') as f:
                data = pickle.load(f)
                self.total_iterations = data
        except:
            print('couldnt load past data')
        
    def _save_iterations(self):
        save_to_file('Game/iterations_counter.bz2', self.total_iterations)

    def _load_past_data(self):
        """
        Internal function that loads the past training data from the previous training run, if it exists
        This ensures that the model always has a large set of training data to avoid overfitting when 
        stopping and restarting the training process.
        """
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
        start = time.time()
        result = self.model.predict(model_input, verbose=0)
        end = time.time()
        self.eval_time += (end-start)

        list_ps = result[0]
        list_vs = result[1]
        result_zip = zip(list_ps, list_vs)

        # go through the result list and send each individual result back to the corresponding process
        for input_repr, key, (p, v) in zip(model_input, self.list_uid, result_zip):
            #self.evaluations[input_repr.data.tobytes()] = (p, v)
            out_queue = self.output_queues[key]
            out_queue.put((p, v))

        # empty the now processed input lists
        self.list_uid = []
        self.list_states = []

    def _split_data(self):
        """
        Internal function that takes the current list of games in the buffer and distributes the game data
        to the training and the test sets that make up the training data class. This ensures that the test set data
        is never used as training data.
        """
        list_states = []
        list_outputs = []
        for game in self.buffer:
            for (state, dist, v) in game:
                list_states.append(state[0])
                list_outputs.append((np.array(dist).flatten(), v))
        X_train, X_test, y_train, y_test = train_test_split(list_states, list_outputs,
                                                            shuffle=True,
                                                            train_size=train_split)
        print(len(X_train), len(X_test))
        self.training_data.add(X_train, X_test, y_train, y_test)
        print(len(self.training_data.X_train))
        self.buffer.clear()

    def _train_network(self):
        """Method to train the network from the accumulated games

        :return: None
        """
        
        dists_train, vs_train = zip(*self.training_data.y_train)
        dists_test, vs_test = zip(*self.training_data.y_test)

        history = self.model.fit(np.array(self.training_data.X_train),
                       [np.array(dists_train), np.array(vs_train)],
                       epochs=epochs,
                       verbose=1,
                       batch_size=batch_size,
                       validation_data=(np.array(self.training_data.X_test),[np.array(dists_test), np.array(vs_test)]),
                       )
        self.total_iterations += 1
        print(time.time()-self.start_time, ' seconds since start')
        print(self.total_iterations, ' training iterations')
        self._save_iterations()

        if self.total_iterations % 20 == 0: # every 2000 games (if 50 games per training), save a model file of this state
            self.model.save(f'model_checkpoint/model_{self.total_iterations}_it.h5')


        self.model.save_weights(checkpoint_path)
        save_to_file('Game/training_data_class.bz2', self.training_data)


def save_to_file(filename, data):
    with bz2.BZ2File(filename, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def h5_to_weights():
    """
    Function that creates a model, loads the .h5 file and saves the weights in the checkpoints path
    """
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        tensorflow.keras.models.load_model('saved_model/model.h5')
    except Exception as e:
        print('EXCEPTION, couldnt load model ', e)
    
    model.save_weights(checkpoint_path)

def weights_to_h5():
    """
    Function that creates a model, loads the weights and saves the model as an .h5 file
    """
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load model ', e)

    model.save('saved_model/model.h5')
