from multiprocessing import Queue, set_start_method
import sys
sys.path.append('..')

from chess import Chessboard
from interface_process import InterfaceProcess

class InterfacePlayer:
    """
    A class representing an interface player
    """
    def __init__(self, chessboard:Chessboard):
        self.chessboard = chessboard
        self.move_queue = Queue()
        #set_start_method('spawn')
        
        interface_process = InterfaceProcess(self.move_queue, self.chessboard)
        #interface_process.daemon = True
        #interface_process.start()
        
        

    def get_move(self):
        move = None
        while not move:
            move = self.move_queue.get()
        return move

    def update(self, move):
        self.move_queue.put(move)