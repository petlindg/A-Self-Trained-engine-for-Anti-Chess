import multiprocessing
import random

from eval_game import EvalGame

class GameProcessEval(multiprocessing.Process):
    """
    A class process to contain and play an EvalGame and communicate results with two different NeuralNetworkProcessEval instances
    """
    def __init__(self, input_queue_1, output_queue_1, input_queue_2, output_queue_2, initial_state, uid):
        """
        A class process to contain and play an EvalGame and communicate results with two different NeuralNetworkProcessEval instances

        :param outgoing_queue_1: The outgoing queue to contain prediction request to a NeuralNetworkProcessEval instance
        :param incoming_queue_1: The incoming queue to contain results from predictions from a NeuralNetworkProcessEval instance
        :param outgoing_queue_2: The outgoing queue to contain prediction request to a NeuralNetworkProcessEval instance
        :param incoming_queue_2: The incoming queue to contain results from predictions from a NeuralNetworkProcessEval instance
        :param initial_state: The Chessboard to play from
        :param uid: identifier of the process
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
        Runs a game
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