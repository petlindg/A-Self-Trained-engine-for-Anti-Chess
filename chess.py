import numpy as np
from enum import IntEnum
from typing import List

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

class Color(IntEnum):
    WHITE = 0
    BLACK = 1

class Piece(IntEnum):
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

    def __init__(self, src_index:np.uint8, dst_index:np.uint8, promotion_type=None):
        self.src_index = src_index
        self.dst_index = dst_index
        self.promotion_type = promotion_type

class Chessboard():

    # main bitboard variable for representing the 12 bitboards.
    # The first index represents color, 0 = white, 1 = black
    # The second index represents piece-type.
    # 0 = pawns, 1 = knights, 2 = bishops, 3 = rooks, 4 = queens, 5 = kings
    bitboards : np.ndarray
    # index 0 = bitboard of combined white pieces, index 1 = bitboard of combined black pieces
    combined  : np.ndarray
    # other logic
    player_to_move : Color

    # init empty board
    def __init__(self):
        self.bitboards = np.zeros((2, 6), dtype=np.uint64)
        self.combined  = np.zeros(2, dtype=np.uint64)
        self.player_to_move = Color.WHITE

    def get_moves(self):

        self.combine_bb()

        moves:List[Move] = []
        moves += self.get_moves_pawns()
        #moves.append(self.get_moves_knights())
        #moves.append(self.get_moves_bishops())
        #moves.append(self.get_moves_rooks())
        #moves.append(self.get_moves_queens())
        moves += self.get_moves_kings()
        return moves

    def get_moves_by_bb(self, src_index:np.uint8, dst_bb:np.uint64):
        moves:List[Move] = []
        dst_index = np.uint8(0)
        bit = np.uint8(1)
        while dst_bb:
            if dst_bb & bit:
                moves.append(Move(src_index, dst_index))
            dst_index += bit
            dst_bb >>= bit
        return moves

    def combine_bb(self):
        index = 0
        for player in self.bitboards:
            combined = np.uint64(0)
            for bb in player:
                combined = np.bitwise_or(combined, bb)
            self.combined[index] = combined
            index += 1

    def get_moves_pawns(self):
        if self.player_to_move:
            return self.get_moves_pawns_black()
        else:
            return self.get_moves_pawns_white()

    def get_moves_pawns_white(self):
        moves:List[Move] = []
        bit = np.uint8(1)
        index = np.uint8(0)
        bb = self.bitboards[0, 0]
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_pawns_white_by_square(index)
            index += bit
            bb = np.right_shift(bb, bit)
        return moves

    def get_moves_pawns_white_by_square(self, src_index:np.uint8):
        bb_moves = np.uint64(0)
        bb_takes = np.uint64(0)

        src_bb = np.left_shift(np.uint64(1), src_index)

        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.left_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_H_BB)))
        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.left_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_A_BB)))
        bb_takes = np.bitwise_and(bb_takes, self.combined[1])

        # only need to return take moves
        if bb_takes:
            return self.get_moves_by_bb(src_index, bb_takes)
        
        bb_moves = np.bitwise_or(bb_moves, np.left_shift(src_bb, np.uint8(8)))
        if (8 <= src_index < 16):
            bb_moves = np.bitwise_or(bb_moves, np.left_shift(src_bb, np.uint8(16)))
        bb_moves = np.bitwise_and(bb_moves, np.bitwise_not(self.combined[0]))

        bb = np.bitwise_or(bb_moves, bb_takes)

        return self.get_moves_by_bb(src_index, bb)

    def get_moves_pawns_black(self):
        moves:List[Move] = []
        bit = np.uint8(1)
        index = np.uint8(0)
        bb = self.bitboards[1,0]
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_pawns_black_by_square(index)
            index += 1
            bb = np.right_shift(bb, bit)
        return moves

    def get_moves_pawns_black_by_square(self, src_index:np.uint8):
        bb_moves = np.uint64(0)
        bb_takes = np.uint64(0)

        src_bb = np.left_shift(np.uint64(1), src_index)

        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.right_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_H_BB)))
        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.right_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_A_BB)))
        bb_takes = np.bitwise_and(bb_takes, self.combined[0])

        # only need to return take moves
        if bb_takes:
            return self.get_moves_by_bb(src_index, bb_takes)

        bb_moves = np.bitwise_or(bb_moves, np.right_shift(src_bb, np.uint8(8)))
        if (48 <= src_index < 56):
            bb_moves = np.bitwise_or(bb_moves, np.right_shift(src_bb, np.uint8(16)))
        bb_moves = np.bitwise_and(bb_moves, np.bitwise_not(self.combined[1]))

        bb = np.bitwise_or(bb_moves, bb_takes)

        return self.get_moves_by_bb(src_index, bb)

    def get_moves_kings(self):
        moves:List[Move] = []
        index = np.uint8(0)
        bit = np.uint8(1)

        bb = self.bitboards[self.player_to_move,5]
        
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_kings_by_square(index)
            index += bit
            bb = np.right_shift(bb, bit)
        return moves

    def get_moves_kings_by_square(self, src_index:np.uint8):
        src_bb = np.left_shift(np.uint64(1), src_index)
        dst_bb = np.uint64(0)
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and((np.left_shift( src_bb, np.uint8(9))), np.bitwise_not(RANK_H_BB)))
        dst_bb = np.bitwise_or(dst_bb,                 np.left_shift( src_bb, np.uint8(8)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and((np.left_shift( src_bb, np.uint8(7))), np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and((np.left_shift( src_bb, np.uint8(1))), np.bitwise_not(RANK_H_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and((np.right_shift(src_bb, np.uint8(1))), np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and((np.right_shift(src_bb, np.uint8(7))), np.bitwise_not(RANK_H_BB)))
        dst_bb = np.bitwise_or(dst_bb,                 np.right_shift(src_bb, np.uint8(8)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and((np.right_shift(src_bb, np.uint8(9))), np.bitwise_not(RANK_A_BB)))

        dst_bb &= ~self.combined[self.player_to_move]

        return self.get_moves_by_bb(src_index, dst_bb)

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
        self.bitboards[0, 0] = np.uint64(0b1111111100000000)
        self.bitboards[0, 1] = np.uint64(0b01000010)
        self.bitboards[0, 2] = np.uint64(0b00100100)
        self.bitboards[0, 3] = np.uint64(0b10000001)
        self.bitboards[0, 4] = np.uint64(0b00010000)
        self.bitboards[0, 5] = np.uint64(0b00001000)
        # black pieces
        self.bitboards[1, 0]    = np.uint64(0b11111111000000000000000000000000000000000000000000000000)
        self.bitboards[1, 1]    = np.uint64(0b0100001000000000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 2]    = np.uint64(0b0010010000000000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 3]    = np.uint64(0b1000000100000000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 4]    = np.uint64(0b0001000000000000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 5]    = np.uint64(0b0000100000000000000000000000000000000000000000000000000000000000)

    def init_board_test_1(self):
        # white pieces
        self.bitboards[0, 3] = np.uint64(0b10000000)
        self.bitboards[0, 5] = np.uint64(0b1100000001000000)
        # black pieces
        self.bitboards[1, 3] = np.uint64(0b0000000100000000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 5] = np.uint64(0b0000001000000011000000000000000000000000000000000000000000000000)
    def init_board_test_2(self):
        # white pieces
        self.bitboards[0, 3] = np.uint64(0b0001000000000000)
        self.bitboards[0, 5] = np.uint64(0b001110000000000000000000)
        # black pieces
        self.bitboards[1, 3] = np.uint64(0b00010000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 5] = np.uint64(0b001110000000000000000000000000000000000000000000)
    def init_board_test_3(self):
        # white pieces
        self.bitboards[0, 3] = np.uint64(0b000100000000000000000000)
        self.bitboards[0, 5] = np.uint64(0b00111000000000000000000000000000)
        # black pieces
        self.bitboards[1, 3] = np.uint64(0b000100000000000000000000000000000000000000000000)
        self.bitboards[1, 5] = np.uint64(0b0011100000000000000000000000000000000000)
    def init_board_test_4(self):
        # white pieces
        self.bitboards[0, 3] = np.uint64(0b1)
        # black pieces
        self.bitboards[1, 5] = np.uint64(0b1000000000000000000000000000000000000000000000000000000000000000)
    def init_board_test_5(self):
        # white pieces
        self.bitboards[0, 0] = np.uint64(0b0000011100000000)
        self.bitboards[0, 5] = np.uint64(0b00000111)
        # black pieces
        self.bitboards[1, 0] = np.uint64(0b11100000000000000000000000000000000000000000000000000000)
        self.bitboards[1, 5] = np.uint64(0b1110000000000000000000000000000000000000000000000000000000000000)
        for pt in self.bitboards[0, :]:
            self.combined[0] |= pt

def print_bb(bb:np.uint64):
    m = np.zeros((8,8), dtype=np.uint8)
    index = np.uint8(0)
    for i in range(8):
        for j in range(8):
            m[7-i, 7-j] = np.bitwise_and(bb, np.uint8(1))
            index += 1
    
    for i in range(8):
        for j in range(8):
            print(str(m[i, j]) + " ", end="")
        print()


def main():
    chessboard = Chessboard()
    chessboard.init_board_standard()
    chessboard.translate_board()