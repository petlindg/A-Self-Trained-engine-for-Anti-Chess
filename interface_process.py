from multiprocessing import Process, Queue
from interface import ChessboardGUI
from chess.chessboard import Chessboard
from chess.move import Move

class InterfaceProcess(Process):
    chessboard:Chessboard
    move_queue:Queue

    def move(self, move:Move):
        self.move_queue.put(move)
        self.chessboard.move(move)
        
    def get_move(self):
        move = self.move_queue.get()
        self.chessboard.move(move)
        return self.chessboard.get()

    def __init__(self, move_queue:Queue, chessboard:Chessboard):
        self.move_queue = move_queue
        self.chessboard = chessboard
        self.interface = ChessboardGUI(send_move=self.move, get_bitboards=self.get_move)

    def run(self):
        self.interface.mainloop()