from multiprocessing import Queue, set_start_method
import sys

from interface_process import InterfaceProcess
sys.path.append('..')

class InterfacePlayer:
    """
    A class representing an interface player
    """
    def __init__(self, chessboard):
        self.chessboard = chessboard
        self.move_queue = Queue()
        set_start_method('spawn')
        
        interface_process = InterfaceProcess(self.move_queue, self.chessboard)
        interface_process.daemon = True
        interface_process.start()
        
        

    def get_move(self):
        return self.move_queue.get()

    def update(self, move):
        self.move_queue.put(move)