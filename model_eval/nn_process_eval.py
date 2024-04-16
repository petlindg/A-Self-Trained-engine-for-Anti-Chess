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


class NeuralNetworkProcessEval(multiprocessing.Process):
    """
    Class that represents a single neural network process, it handles incoming requests from
    the game processes and manages the training and execution of the neural network.
    """
    def __init__(self, input_queue, output_queues, nn_batch:int, model=None):
        super(NeuralNetworkProcessEval, self).__init__()
        self.input_queue = input_queue # shared queue for the input
        self.output_queues = output_queues # dict of queues to send back, has UID as key and queue as value
        self.list_uid = []
        self.list_states = []
        self.batch_size = nn_batch
        self.finished_counter = 0

        self.model = model

    def run(self):
        """Start the neural network process allowing it to process data

        :return: None
        """

        while True:
            # get the latest request from the queue
            # based on the request type, perform some action
            request_type, uid, data = self.input_queue.get()
            # if the request is an evaluation then store the data for later evaluation
            # also check if the data already exists in the dictionary, if it does,
            # return the value to the process which asked for it
            if request_type == "finished":
                self.finished_counter += 1
                print(data)
            else:
                self.list_uid.append(uid)
                self.list_states.append(data)

            # if the list of pending evaluation states is large enough
            # perform the model evaluation on the list
            if len(self.list_states) >= self.batch_size or self.finished_counter +len(self.list_states) == batch_size*2:
                self._process_requests()
        
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
            out_queue = self.output_queues[key]
            out_queue.put((p, v))

        # empty the now processed input lists
        self.list_uid = []
        self.list_states = []