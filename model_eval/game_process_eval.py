import multiprocessing
import random

from eval_game import EvalGame

class GameProcessEval(multiprocessing.Process):
    """
    Class representing a single process that plays games using the neural network. 
    This class communicates with and sends requests to the neural network process and recieves evaluations from the 
    neural network.
    """
    def __init__(self, input_queue_1, output_queue_1, input_queue_2, output_queue_2, initial_state, uid):
        """
        The game process class has two queues, the incoming and the outcoming queue
        the incoming queue is the data that the game process recieves from the neural network
        and the outgoing queue is the queue that the game process puts data on, for the network process to handle
        the uid is the unique identifier for the process, it is used for identification purposes. 
        """
        super(GameProcessEval, self).__init__()
        self.outgoing_queue_1 = input_queue_1
        self.incoming_queue_1 = output_queue_1
        self.outgoing_queue_2 = input_queue_2
        self.incoming_queue_2 = output_queue_2
        self.initial_state = initial_state
        self.uid = uid

    def run(self):
        """
        Function that continually plays games by sending requests to the neural network
        """
        from chess.chessboard import Chessboard
        from Game.TrainingGame import TrainingGame
        # TODO re-enable random states once that is on main
        #from Game.state_generator import generate_random_state

        random.seed()
        # while the process is running, keep running training games
        game = EvalGame(initial_state=self.initial_state,
                        outgoing_queue_1=self.outgoing_queue_1,
                        incoming_queue_1=self.incoming_queue_1,
                        outgoing_queue_2=self.outgoing_queue_2,
                        incoming_queue_2=self.incoming_queue_2,
                        uid=self.uid)
        
        result = game.run()

        self.outgoing_queue_1.put(('finished', self.uid, result))
        self.outgoing_queue_2.put(('finished', self.uid, result))