import numpy as np
import enum

# ranks used for masking
# ranks by number
RANK_8_BB = 0b1111111100000000000000000000000000000000000000000000000000000000
RANK_7_BB = 0b11111111000000000000000000000000000000000000000000000000
RANK_6_BB = 0b111111110000000000000000000000000000000000000000
RANK_5_BB = 0b1111111100000000000000000000000000000000
RANK_4_BB = 0b11111111000000000000000000000000
RANK_3_BB = 0b111111110000000000000000
RANK_2_BB = 0b1111111100000000
RANK_1_BB = 0b11111111

# ranks by letter
RANK_A_BB = 0b1000000010000000100000001000000010000000100000001000000010000000
RANK_B_BB = 0b0100000001000000010000000100000001000000010000000100000001000000
RANK_C_BB = 0b0010000000100000001000000010000000100000001000000010000000100000
RANK_D_BB = 0b0001000000010000000100000001000000010000000100000001000000010000
RANK_E_BB = 0b0000100000001000000010000000100000001000000010000000100000001000
RANK_F_BB = 0b0000010000000100000001000000010000000100000001000000010000000100
RANK_G_BB = 0b0000001000000010000000100000001000000010000000100000001000000010
RANK_H_BB = 0b0000000100000001000000010000000100000001000000010000000100000001

class Color(enum):
    WHITE = 0
    BLACK = 1

class Piece(enum):
    PAWN   = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK   = 3
    QUEEN  = 4
    KING   = 5

class Move():
    src_index      : np.uint8
    dst_index      : np.uint8
    promotion_type : np.Piece

    def __init__(self, src_index:np.uint8, dst_index:np.uint8, promotion_type):
        self.src_index = src_index
        self.dst_index = dst_index
        self.promotion_type = promotion_type

class Chessboard():
    # white pieces
    white_pawns    : np.uint64
    white_knights  : np.uint64
    white_bishops  : np.uint64
    white_rooks    : np.uint64
    white_queens   : np.uint64
    white_kings    : np.uint64
    white_combined : np.uint64
    # black pieces
    black_pawns    : np.uint64
    black_knights  : np.uint64
    black_bishops  : np.uint64
    black_rooks    : np.uint64
    black_queens   : np.uint64
    black_kings    : np.uint64
    black_combined : np.uint64
    # other logic
    player_to_move : Color

    # init empty board
    def __init__(self):
        # white pieces
        self.white_pawns = np.uint64(0)
        self.white_knights = np.uint64(0)
        self.white_bishops = np.uint64(0)
        self.white_rooks = np.uint64(0)
        self.white_queens = np.uint64(0)
        self.white_kings = np.uint64(0)
        self.white_combined = np.uint64(0)
        # black pieces
        self.black_pawns = np.uint64(0)
        self.black_knights = np.uint64(0)
        self.black_bishops = np.uint64(0)
        self.black_rooks = np.uint64(0)
        self.black_queens = np.uint64(0)
        self.black_kings = np.uint64(0)
        self.black_combined = np.uint64(0)

        self.bitboards = np.zeros((2, 6), dtype=np.uint64)
        
        self.white_to_move = True

    def get_moves_king(self, bb:np.uint64, bb_color_combined):
        north_west = (bb << 9) & ~RANK_H_BB
        north      = bb << 8
        north_east = (bb << 7) & ~RANK_A_BB
        west       = (bb << 1) & ~RANK_H_BB
        east       = (bb >> 1) & ~RANK_A_BB
        south_west = (bb >> 7) & ~RANK_H_BB
        south      = bb >> 8
        south_east = (bb >> 9) & ~RANK_A_BB

        move_bb = north_west | north | north_east | west | east | south_west | south | south_east
        move_bb &= ~bb_color_combined
        return move_bb

    # init standard chess board
    def init_board_standard(self):

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
    def init_board_test_1(self):
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
    def init_board_test_2(self):
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
    def init_board_test_3(self):
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
    def init_board_test_4(self):
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
