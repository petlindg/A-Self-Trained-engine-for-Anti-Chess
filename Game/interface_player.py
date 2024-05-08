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
        print("starting process")
        interface_process = InterfaceProcess(self.move_queue, self.chessboard)
        print("process started")
        interface_process.daemon = True
        interface_process.start()
        
        

    def get_move(self):
        print("waiting for move")
        move = None
        while not move:
            move = self.move_queue.get()
        print("got move")
        return move

    def update(self, move):
        self.move_queue.put(move)