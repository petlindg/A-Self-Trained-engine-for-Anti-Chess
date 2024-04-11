import multiprocessing
import random
import threading
import config
import queue
threads = config.threads
"""
===============================
Protocol format for the queues:
===============================


thread to process:
(request_type, tuid, data)

process to network:
(request_type, tuid, pid, data)

network to process:
(tuid, data)

process to thread:
(data)


"""

class GameProcess(multiprocessing.Process):
    """
    Class representing a single process that plays games using the neural network. 
    This class communicates with and sends requests to the neural network process and recieves evaluations from the 
    neural network.
    """
    def __init__(self, nn_queue, process_queue, initial_state, uid):
        """
        The game process class has two queues, the incoming and the outcoming queue
        the incoming queue is the data that the game process recieves from the neural network
        and the outgoing queue is the queue that the game process puts data on, for the network process to handle
        the uid is the unique identifier for the process, it is used for identification purposes. 
        """
        super(GameProcess, self).__init__()
        # process level variables
        self.nn_queue = nn_queue # the queue where we send data to the neural network
        self.process_queue = process_queue # the queue where we recieve data from the neural network to the process
        self.initial_state = initial_state
        self.uid = uid
        # thread level variables (Note that the queue class is different)
        self.t_outgoing_queues = {} # the dict of queues for every thread where it sends data
        self.threads = [] 
        self.t_incoming_queue = multiprocessing.Queue() # the queue where it recieves the data from all the threads

    def _setup_process(self):
        for i in range(0, threads):
            # Note that this is not the same as multiprocessing Queue, but rather the threading variant of queue.
            t_queue = multiprocessing.Queue()
            thread = GameThread(
                t_incoming_queue=t_queue,
                t_outgoing_queue=self.t_incoming_queue,
                initial_state=self.initial_state,
                tuid=i
            )
            self.t_outgoing_queues[i] = t_queue
            self.threads.append(thread)

        for thread in self.threads:
            thread.daemon = True
            thread.start()
        print(len(self.threads))

    def _process_incoming(self, request):
        "transfer the data from the network to the specific thread"
        tuid, p, v = request
        #print(f'{self.uid} processed {tuid} v: {round(v[0], 2)}')
        self.t_outgoing_queues[tuid].put((p, v))

    def _process_outbound(self, request):
        "transfer the data from the thread to the network"
        request_type, tuid, data = request
        self.nn_queue.put((request_type, tuid, self.uid, data))

    def run(self):
        """
        Function that continually plays games by sending requests to the neural network
        """
        self._setup_process()        
        while True:
            # check if there is any inbound data from the network
            try:
                result = self.process_queue.get(False)
                self._process_incoming(result)
            except queue.Empty:
                pass

            # check if there is any inbound data from the threads
            try:
                result = self.t_incoming_queue.get(False)
                self._process_outbound(result)
            except queue.Empty:
                pass



        


class GameThread(threading.Thread):
    def __init__(self,t_incoming_queue, t_outgoing_queue, initial_state, tuid):
        super(GameThread, self).__init__()
        self.uid = tuid
        self.t_outgoing_queue = t_outgoing_queue # the queue where we send data to the process
        self.t_incoming_queue = t_incoming_queue # the queue where we recieve data from the process
        self.initial_state = initial_state

    def run(self):
        """
        Function that continually plays games by sending requests to the neural network
        """
        from chess import Chessboard
        from Game.TrainingGame import TrainingGame
        from Game.state_generator import generate_random_state

        random.seed()
        # while the process is running, keep running training games
        while True:
            self.chessboard = Chessboard(self.initial_state)

            random_state = generate_random_state(config.piece_list)
            if config.random_state_generation:
                game = TrainingGame(initial_state=Chessboard(random_state), outgoing_queue=self.t_outgoing_queue,
                                    incoming_queue=self.t_incoming_queue, uid=self.uid)
            else:
                game = TrainingGame(initial_state=self.chessboard, outgoing_queue=self.t_outgoing_queue,
                                    incoming_queue=self.t_incoming_queue, uid=self.uid)
            result = game.run()

            self.t_outgoing_queue.put(('finished', self.uid, (game.get_history(), result)))