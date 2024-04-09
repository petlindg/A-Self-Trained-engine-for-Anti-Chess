import multiprocessing
import random

import config

class GameProcess(multiprocessing.Process):
    """
    Class representing a single process that plays games using the neural network. 
    This class communicates with and sends requests to the neural network process and recieves evaluations from the 
    neural network.
    """
    def __init__(self, input_queue, output_queue, initial_state, uid):
        """
        The game process class has two queues, the incoming and the outcoming queue
        the incoming queue is the data that the game process recieves from the neural network
        and the outgoing queue is the queue that the game process puts data on, for the network process to handle
        the uid is the unique identifier for the process, it is used for identification purposes. 
        """
        super(GameProcess, self).__init__()
        self.outgoing_queue = input_queue
        self.incoming_queue = output_queue
        self.initial_state = initial_state
        self.uid = uid

    def run(self):
        """
        Function that continually plays games by sending requests to the neural network
        """
        from chess.chessboard import Chessboard
        from Game.TrainingGame import TrainingGame
        from Game.state_generator import generate_random_state

        random.seed()
        # while the process is running, keep running training games
        while True:
            self.chessboard = Chessboard(self.initial_state)

            random_state = generate_random_state(config.piece_list)
            if config.random_state_generation:
                game = TrainingGame(initial_state=Chessboard(random_state), outgoing_queue=self.outgoing_queue,
                                    incoming_queue=self.incoming_queue, uid=self.uid)
            else:
                game = TrainingGame(initial_state=self.chessboard, outgoing_queue=self.outgoing_queue,
                                    incoming_queue=self.incoming_queue, uid=self.uid)
            result = game.run()

            self.outgoing_queue.put(('finished', self.uid, (game.get_history(), result)))