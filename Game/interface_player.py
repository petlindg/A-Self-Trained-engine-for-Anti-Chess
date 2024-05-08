from multiprocessing import Queue, set_start_method
import sys
sys.path.append('..')

from chess.chessboard import Chessboard
from interface_process import InterfaceProcess

class InterfacePlayer:
    """
    A class representing an interface player
    """
    def __init__(self, chessboard:Chessboard):
        set_start_method('spawn')
        self.chessboard = chessboard
        self.outgoing_queue = Queue()
        self.incoming_queue = Queue()
        interface_process = InterfaceProcess(incoming_queue=self.incoming_queue,
                                             outgoing_queue=self.outgoing_queue,
                                             chessboard=self.chessboard)
        interface_process.daemon = True
        interface_process.start()
        
        

    def get_move(self):
        move = self.outgoing_queue.get()
        self.outgoing_queue.empty()
        return move

    def update(self, move):
        self.chessboard.move(move)
        self.incoming_queue.put(move)