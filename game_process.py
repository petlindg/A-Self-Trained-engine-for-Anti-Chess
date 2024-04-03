import multiprocessing
import random

import config

class GameProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, initial_state, uid):
        super(GameProcess, self).__init__()
        self.outgoing_queue = input_queue
        self.incoming_queue = output_queue
        self.initial_state = initial_state
        self.uid = uid

    def run(self):
        from chess import Chessboard
        from Game.TrainingGame import TrainingGame
        from Game.state_generator import generate_random_state

        random.seed(self.uid)
        # while the process is running, keep running training games
        while True:
            chessboard = Chessboard(self.initial_state)
            random_state = generate_random_state(config.piece_list)
            if config.random_state_generation:
                game = TrainingGame(initial_state=Chessboard(random_state), outgoing_queue=self.outgoing_queue,
                                    incoming_queue=self.incoming_queue, uid=self.uid)
            else:
                game = TrainingGame(initial_state=chessboard, outgoing_queue=self.outgoing_queue,
                                    incoming_queue=self.incoming_queue, uid=self.uid)
            result = game.run()
            print(result)
            self.outgoing_queue.put(('finished', self.uid, game.get_history()))