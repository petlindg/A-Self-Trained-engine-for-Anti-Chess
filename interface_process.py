from copy import deepcopy
from multiprocessing import Process, Queue
from interface import ChessboardGUI, WINDOW_SIZE
from chess import Chessboard, Move

class InterfaceProcess(Process):
    chessboard:Chessboard
    incoming_queue:Queue
    outgoing_queue:Queue

    def move(self, move:Move):
        if(self.chessboard.is_valid_move(move)):
            self.outgoing_queue.put(move)
            return True
        else:
            return False
        
    def get_move(self):
        move = self.incoming_queue.get()
        self.chessboard.move(move)
        return self.chessboard.get()

    def __init__(self, incoming_queue:Queue, outgoing_queue:Queue, chessboard:Chessboard):
        super(InterfaceProcess, self).__init__()
        self.incoming_queue = incoming_queue
        self.outgoing_queue = outgoing_queue
        self.chessboard = chessboard

    def run(self):
        self.interface = ChessboardGUI(size=WINDOW_SIZE, send_move=self.move, get_bitboards=self.get_move)
        self.interface.init_board(self.chessboard.get())
        self.interface.mainloop()