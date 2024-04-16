import multiprocessing
import numpy as np

from config import batch_size

class NeuralNetworkProcessEval(multiprocessing.Process):
    """
    Class that represents a single neural network process, it handles incoming requests from
    the game processes and manages the execution of the neural network.
    """
    def __init__(self, input_queue, output_queues, nn_batch:int, model):
        """
        Class that represents a single neural network process, it handles incoming requests from
        the game processes and manages the training and execution of the neural network.

        :param outgoing_queue: The outgoing queue to contain results from a prediction to a GameProcessEval instance
        :param incoming_queue: The incoming queue to contain prediction request from a GameProcessEval instance
        :param nn_batch: The number of data entries to predict at a time
        :param model: The model to predict from
        """
        super(NeuralNetworkProcessEval, self).__init__()
        self.input_queue = input_queue # shared queue for the input
        self.output_queues = output_queues # dict of queues to send back, has UID as key and queue as value
        self.list_uid = []
        self.list_states = []
        self.batch_size = nn_batch
        self.finished_counter = 0
        self.total_games = nn_batch*2

        self.model = model

    def run(self):
        """
        Start the neural network process allowing it to process data
        """

        # keep process alive while games are in progress
        while self.finished_counter < self.total_games:
            request_type, uid, data = self.input_queue.get()
            if request_type == "finished":
                self.finished_counter += 1
                print(data)
            else:
                self.list_uid.append(uid)
                self.list_states.append(data)

            # if the list of pending evaluation states is large enough
            # perform the model evaluation on the list
            if len(self.list_states) >= self.batch_size or self.finished_counter + len(self.list_states) == self.total_games:
                self._process_requests()
        
        print("All games complete")
        
    def _process_requests(self):
        """
        Method to process the requested evaluations and return the results to all the linked processes.
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