import numpy as np

class Chessboard:

    def __init__(self):
        # white pieces
        self.white_pawns = np.uint64(0)
        self.white_knights = np.uint64(0)
        self.white_bishops = np.uint64(0)
        self.white_rooks = np.uint64(0)
        self.white_queens = np.uint64(0)
        self.white_kings = np.uint64(0)
        # black pieces
        self.black_pawns = np.uint64(0)
        self.black_knights = np.uint64(0)
        self.black_bishops = np.uint64(0)
        self.black_rooks = np.uint64(0)
        self.black_queens = np.uint64(0)
        self.black_kings = np.uint64(0)
        
        self.bitboards = np.zeros((2, 6), dtype=np.uint64)
        
        self.white_to_move = True
    def initStandardBoard(self):

        # white pieces
        self.white_pawns    = 0b1111111100000000
        self.white_knights  = 0b01000010
        self.white_bishops  = 0b00100100
        self.white_rooks    = 0b10000001
        self.white_queens   = 0b00010000
        self.white_kings    = 0b00001000
        # black pieces
        self.black_pawns    = 0b11111111000000000000000000000000000000000000000000000000
        self.black_knights  = 0b0100001000000000000000000000000000000000000000000000000000000000
        self.black_bishops  = 0b0010010000000000000000000000000000000000000000000000000000000000
        self.black_rooks    = 0b1000000100000000000000000000000000000000000000000000000000000000
        self.black_queens   = 0b0001000000000000000000000000000000000000000000000000000000000000
        self.black_kings    = 0b0000100000000000000000000000000000000000000000000000000000000000

        self.bitboards[0, 0] = self.white_pawns
        self.bitboards[0, 1] = self.white_knights
        self.bitboards[0, 2] = self.white_bishops
        self.bitboards[0, 3] = self.white_rooks
        self.bitboards[0, 4] = self.white_queens
        self.bitboards[0, 5] = self.white_kings
        self.bitboards[1, 0] = self.black_pawns
        self.bitboards[1, 1] = self.black_knights
        self.bitboards[1, 2] = self.black_bishops
        self.bitboards[1, 3] = self.black_rooks
        self.bitboards[1, 4] = self.black_queens
        self.bitboards[1, 5] = self.black_kings

    def initTestBoard1(self):
        # white pieces
        self.white_rooks    = 0b10000000
        self.white_kings    = 0b1100000001000000
        # black pieces
        self.black_rooks    = 0b0000000100000000000000000000000000000000000000000000000000000000
        self.black_kings    = 0b0000001000000011000000000000000000000000000000000000000000000000

        self.bitboards[0, 0] = self.white_pawns
        self.bitboards[0, 1] = self.white_knights
        self.bitboards[0, 2] = self.white_bishops
        self.bitboards[0, 3] = self.white_rooks
        self.bitboards[0, 4] = self.white_queens
        self.bitboards[0, 5] = self.white_kings
        self.bitboards[1, 0] = self.black_pawns
        self.bitboards[1, 1] = self.black_knights
        self.bitboards[1, 2] = self.black_bishops
        self.bitboards[1, 3] = self.black_rooks
        self.bitboards[1, 4] = self.black_queens
        self.bitboards[1, 5] = self.black_kings

    def initTestBoard2(self):
        # white pieces
        self.white_rooks    = 0b0001000000000000
        self.white_kings    = 0b001110000000000000000000
        # black pieces
        self.black_rooks    = 0b00010000000000000000000000000000000000000000000000000000
        self.black_kings    = 0b001110000000000000000000000000000000000000000000

        self.bitboards[0, 0] = self.white_pawns
        self.bitboards[0, 1] = self.white_knights
        self.bitboards[0, 2] = self.white_bishops
        self.bitboards[0, 3] = self.white_rooks
        self.bitboards[0, 4] = self.white_queens
        self.bitboards[0, 5] = self.white_kings
        self.bitboards[1, 0] = self.black_pawns
        self.bitboards[1, 1] = self.black_knights
        self.bitboards[1, 2] = self.black_bishops
        self.bitboards[1, 3] = self.black_rooks
        self.bitboards[1, 4] = self.black_queens
        self.bitboards[1, 5] = self.black_kings

    def initTestBoard3(self):
        # white pieces
        self.white_rooks    = 0b000100000000000000000000
        self.white_kings    = 0b00111000000000000000000000000000
        # black pieces
        self.black_rooks    = 0b000100000000000000000000000000000000000000000000
        self.black_kings    = 0b0011100000000000000000000000000000000000

        self.bitboards[0, 0] = self.white_pawns
        self.bitboards[0, 1] = self.white_knights
        self.bitboards[0, 2] = self.white_bishops
        self.bitboards[0, 3] = self.white_rooks
        self.bitboards[0, 4] = self.white_queens
        self.bitboards[0, 5] = self.white_kings
        self.bitboards[1, 0] = self.black_pawns
        self.bitboards[1, 1] = self.black_knights
        self.bitboards[1, 2] = self.black_bishops
        self.bitboards[1, 3] = self.black_rooks
        self.bitboards[1, 4] = self.black_queens
        self.bitboards[1, 5] = self.black_kings

    def initTestBoard4(self):
        # white pieces
        self.white_rooks    = 0b1
        # black pieces
        self.black_kings    = 0b1000000000000000000000000000000000000000000000000000000000000000

        self.bitboards[0, 0] = self.white_pawns
        self.bitboards[0, 1] = self.white_knights
        self.bitboards[0, 2] = self.white_bishops
        self.bitboards[0, 3] = self.white_rooks
        self.bitboards[0, 4] = self.white_queens
        self.bitboards[0, 5] = self.white_kings
        self.bitboards[1, 0] = self.black_pawns
        self.bitboards[1, 1] = self.black_knights
        self.bitboards[1, 2] = self.black_bishops
        self.bitboards[1, 3] = self.black_rooks
        self.bitboards[1, 4] = self.black_queens
        self.bitboards[1, 5] = self.black_kings

    def get_player_to_move(self):
        '''This method '''
        if self.white_to_move:
            return 'White'
        else:
            return 'Black'
    
    def update_player_to_move(self):
        if self.white_to_move:
            self.white_to_move = False       
        else:
            self.white_to_move = True
    
    def make_move(self, move):

        pass