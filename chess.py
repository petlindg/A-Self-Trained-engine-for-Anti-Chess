import numpy as np
from enum import Enum

# ranks used for masking
# ranks by number
RANK_8_BB = np.uint64(0b1111111100000000000000000000000000000000000000000000000000000000)
RANK_7_BB = np.uint64(0b11111111000000000000000000000000000000000000000000000000)
RANK_6_BB = np.uint64(0b111111110000000000000000000000000000000000000000)
RANK_5_BB = np.uint64(0b1111111100000000000000000000000000000000)
RANK_4_BB = np.uint64(0b11111111000000000000000000000000)
RANK_3_BB = np.uint64(0b111111110000000000000000)
RANK_2_BB = np.uint64(0b1111111100000000)
RANK_1_BB = np.uint64(0b11111111)

rank_nr_list: list[np.uint64] = [RANK_8_BB, RANK_7_BB, RANK_6_BB, RANK_5_BB, RANK_4_BB, RANK_3_BB, RANK_2_BB, RANK_1_BB]

# ranks by letter
RANK_A_BB = np.uint64(0b1000000010000000100000001000000010000000100000001000000010000000)
RANK_B_BB = np.uint64(0b0100000001000000010000000100000001000000010000000100000001000000)
RANK_C_BB = np.uint64(0b0010000000100000001000000010000000100000001000000010000000100000)
RANK_D_BB = np.uint64(0b0001000000010000000100000001000000010000000100000001000000010000)
RANK_E_BB = np.uint64(0b0000100000001000000010000000100000001000000010000000100000001000)
RANK_F_BB = np.uint64(0b0000010000000100000001000000010000000100000001000000010000000100)
RANK_G_BB = np.uint64(0b0000001000000010000000100000001000000010000000100000001000000010)
RANK_H_BB = np.uint64(0b0000000100000001000000010000000100000001000000010000000100000001)

rank_l_list: list[np.uint64] = [RANK_A_BB, RANK_B_BB, RANK_C_BB, RANK_D_BB, RANK_E_BB, RANK_F_BB, RANK_G_BB, RANK_H_BB]

class Color(Enum):
    WHITE = 0
    BLACK = 1

class Piece(Enum):
    PAWN   = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK   = 3
    QUEEN  = 4
    KING   = 5

class Move():
    src_index      : np.uint8
    dst_index      : np.uint8
    promotion_type : Piece

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

    def translate_board(self):
        # The outer structure is an 8 by 8 grid representing 'squares' of the board
        # the innermost list is meant to be represented as followed: (the 17 planes of the input repr.)
        # [If W pawn, If W knight, If W bishop, If W rook, If W queen, If W king, ...
        #  If B pawn, If B knight, If B bishop, If B rook, If B queen, If B king, ...
        #  Repetition Count for W as Integer, Repetition Count for B as Integer, ...
        #  1 If White, 0 If black (color), ...
        #  No Progress Count as Integer, ...
        #  1 if the grid square is an en passant square, 0 if not
        representation = [[
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []]
        ]]
        # the full shape of the array is in the form of (1, 8, 8, 17)
        # technically not that efficient with these 4 nested for loops, however
        # the total looping is only around 750 loops (2*6*8*8) which is mostly fine
        for player in self.bitboards:
            for piece_type in player:
                for i in range(0, 8):
                    for l in range(0, 8):
                        position = np.bitwise_and(rank_nr_list[i], rank_l_list[l])
                        if np.bitwise_and(piece_type, position) != 0:
                            representation[0][i][l].append(1)
                        else:
                            representation[0][i][l].append(0)

        # TODO represent repetitions in some shape or form for the Chessboard class
        repetitions_w = 0
        repetitions_b = 0
        # TODO represent color for the current players turn
        color = 0
        # TODO represent the no progress counter
        no_progress = 0
        # TODO calculate and represent en passant squares as specific squares in a 8x8 bitboard
        en_passant = np.uint64(0b0000000000000000000000000000000000000000000000000000000000000000)
        for i in range(0, 8):
            for l in range(0, 8):
                representation[0][i][l].append(repetitions_w)
                representation[0][i][l].append(repetitions_b)
                representation[0][i][l].append(color)
                representation[0][i][l].append(no_progress)
                position = np.bitwise_and(rank_nr_list[i], rank_l_list[l])
                if np.bitwise_and(en_passant, position) != 0:
                    representation[0][i][l].append(1)
                else:
                    representation[0][i][l].append(0)


        for row in range(0,8):
            print(representation[0][row])
        print(np.array(representation).shape)



    # init standard chess board
    def init_board_standard(self):

        # white pieces
        self.white_pawns    = np.uint64(0b1111111100000000)
        self.white_knights  = np.uint64(0b01000010)
        self.white_bishops  = np.uint64(0b00100100)
        self.white_rooks    = np.uint64(0b10000001)
        self.white_queens   = np.uint64(0b00010000)
        self.white_kings    = np.uint64(0b00001000)
        # black pieces
        self.black_pawns    = np.uint64(0b11111111000000000000000000000000000000000000000000000000)
        self.black_knights  = np.uint64(0b0100001000000000000000000000000000000000000000000000000000000000)
        self.black_bishops  = np.uint64(0b0010010000000000000000000000000000000000000000000000000000000000)
        self.black_rooks    = np.uint64(0b1000000100000000000000000000000000000000000000000000000000000000)
        self.black_queens   = np.uint64(0b0001000000000000000000000000000000000000000000000000000000000000)
        self.black_kings    = np.uint64(0b0000100000000000000000000000000000000000000000000000000000000000)

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
        self.white_rooks    = np.uint64(0b10000000)
        self.white_kings    = np.uint64(0b1100000001000000)
        # black pieces
        self.black_rooks    = np.uint64(0b0000000100000000000000000000000000000000000000000000000000000000)
        self.black_kings    = np.uint64(0b0000001000000011000000000000000000000000000000000000000000000000)

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
        self.white_rooks    = np.uint64(0b0001000000000000)
        self.white_kings    = np.uint64(0b001110000000000000000000)
        # black pieces
        self.black_rooks    = np.uint64(0b00010000000000000000000000000000000000000000000000000000)
        self.black_kings    = np.uint64(0b001110000000000000000000000000000000000000000000)

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
        self.white_rooks    = np.uint64(0b000100000000000000000000)
        self.white_kings    = np.uint64(0b00111000000000000000000000000000)
        # black pieces
        self.black_rooks    = np.uint64(0b000100000000000000000000000000000000000000000000)
        self.black_kings    = np.uint64(0b0011100000000000000000000000000000000000)

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
        self.white_rooks    = np.uint64(0b1)
        # black pieces
        self.black_kings    = np.uint64(0b1000000000000000000000000000000000000000000000000000000000000000)

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


def main():
    chessboard = Chessboard()
    chessboard.init_board_standard()
    chessboard.translate_board()

main()