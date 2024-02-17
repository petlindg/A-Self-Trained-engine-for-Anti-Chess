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

KNIGHT_BB = np.zeros(64, dtype=np.uint64)
KING_BB = np.zeros(64, dtype=np.uint64)

def knight_bb_init(bbs):
    # Init of movegeneration bitboards for knights.
    # The bitboard of bbs[index] represents the bitboard with all possible destinationsquares given a source square = index
    src_bb = np.uint64(1) # source bb to generate moves from
    for index in range(np.uint64(64)):
        dst_bb = np.uint64(0) # destination bb to track all possible move destinations

        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(17)), np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(15)), np.bitwise_not(RANK_H_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(10)), np.bitwise_not(np.bitwise_or(RANK_A_BB, RANK_B_BB))))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(6)),  np.bitwise_not(np.bitwise_or(RANK_G_BB, RANK_H_BB))))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and( np.left_shift(src_bb, np.uint8(6)),  np.bitwise_not(np.bitwise_or(RANK_A_BB, RANK_B_BB))))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and( np.left_shift(src_bb, np.uint8(10)), np.bitwise_not(np.bitwise_or(RANK_G_BB, RANK_H_BB))))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and( np.left_shift(src_bb, np.uint8(15)), np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and( np.left_shift(src_bb, np.uint8(17)), np.bitwise_not(RANK_H_BB)))

        bbs[index] = dst_bb
        src_bb = np.left_shift(src_bb, np.uint8(1)) # shift src_bb to match index


def king_bb_init(bbs):
    # Init of movegeneration bitboards for kings.
    # The bitboard of bbs[index] represents the bitboard with all possible destinationsquares given a source square = index
    src_bb = np.uint64(1) # source bb to generate moves from
    for index in range(np.uint64(64)):
        dst_bb = np.uint64(0) # # destination bb to track all possible move destinations

        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb,                np.right_shift(src_bb, np.uint8(8)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_H_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.right_shift(src_bb, np.uint8(1)), np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.left_shift(src_bb, np.uint8(1)),  np.bitwise_not(RANK_H_BB)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.left_shift(src_bb, np.uint8(7)),  np.bitwise_not(RANK_A_BB)))
        dst_bb = np.bitwise_or(dst_bb,                np.left_shift(src_bb, np.uint8(8)))
        dst_bb = np.bitwise_or(dst_bb, np.bitwise_and(np.left_shift(src_bb, np.uint8(9)),  np.bitwise_not(RANK_H_BB)))

        bbs[index] = dst_bb
        src_bb = np.left_shift(src_bb, np.uint8(1))# shift src_bb to match index

knight_bb_init(KNIGHT_BB)
king_bb_init(KING_BB)

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

    def print(self):
        # print function for moves
        if self.promotion_type:
            print("src: " + str(self.src_index) + ", dst: " + str(self.dst_index) + ", pro: " + str(self.promotion_type))
        else:
            print("src: " + str(self.src_index) + ", dst: " + str(self.dst_index))

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

    def get(self):
        # used for interface to display
        return self.bitboards

    def try_move(self, src_index, dst_index):
        # used for interface to attempt a move, returns True if successful, False if not successful
        move = Move(np.uint8(src_index), np.uint8(dst_index)) # create function parameters into a Move()
        moves = self.get_moves() # get legal moves
        for m in moves:
            # checks if input move is in legal moves
            if move.src_index == m.src_index and move.dst_index == m.dst_index:
                self.move(move)
                return True
        return False

    def move(self, move:Move):
        # main function to move a piece and updates all nessecary properties of class
        bit = np.uint64(1)
        # create move indexes into bbs
        src_bb = np.left_shift(bit, move.src_index)
        dst_bb = np.left_shift(bit, move.dst_index)

        player = self.player_to_move
        opponent = (self.player_to_move+1)%2

        # iterate over all piece_type bbs and remove all bits on destination square for opponent
        # and add bit on piece_type for player if piece_type has a 1 on source square
        for i in range(6):
            self.bitboards[opponent, i] = np.bitwise_and(self.bitboards[opponent, i], np.bitwise_not(dst_bb))
            if np.bitwise_and(self.bitboards[player, i], src_bb):
                self.bitboards[player, i] = np.bitwise_xor(self.bitboards[player, i], src_bb)
                self.bitboards[player, i] = np.bitwise_or(self.bitboards[player, i], dst_bb)

        # update player to move
        self.player_to_move = (self.player_to_move+1)%2

    def get_moves(self):
        # get legal moves given current state

        # combines the piece_type bitboards of both colors seperately, used in later calculations
        self.combine_bb()

        # var to hold legal moves
        moves:List[Move] = []

        # get moves by different piece_types
        moves += self.get_moves_pawns()
        moves += self.get_moves_knights()
        #moves += self.get_moves_bishops()
        #moves += self.get_moves_rooks()
        #moves += self.get_moves_queens()
        moves += self.get_moves_kings()

        return moves

    def get_moves_by_bb(self, src_index:np.uint8, dst_bb:np.uint64):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = np.uint8(0)
        bit = np.uint8(1)
        while dst_bb:
            if np.bitwise_and(dst_bb, bit):
                moves.append(Move(src_index, dst_index))
            dst_index += bit
            dst_bb = np.right_shift(dst_bb, bit)
        return moves

    def combine_bb(self):
        # combines bb of both playercolors seperately
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
        # get moves by the white pawn bb and returns them as [Move]
        moves:List[Move] = []
        bit = np.uint8(1)
        src_index = np.uint8(0)
        bb = self.bitboards[Color.WHITE, Piece.PAWN]
        # iterate over board, if white pawn on src_index, get pawn moves from that source index
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_pawns_white_by_square(src_index)
            src_index += bit
            bb = np.right_shift(bb, bit)
        return moves

    def get_moves_pawns_white_by_square(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        bb_moves = np.uint64(0)
        bb_takes = np.uint64(0)

        src_bb = np.left_shift(np.uint64(1), src_index)

        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.left_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_H_BB)))
        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.left_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_A_BB)))
        bb_takes = np.bitwise_and(bb_takes, self.combined[1])

        if bb_takes:
            return self.get_moves_by_bb(src_index, bb_takes)
        
        bb_moves = np.bitwise_or(bb_moves, np.left_shift(src_bb, np.uint8(8)))
        if (8 <= src_index < 16):
            bb_moves = np.bitwise_or(bb_moves, np.left_shift(src_bb, np.uint8(16)))
        bb_moves = np.bitwise_and(bb_moves, np.bitwise_not(self.combined[0]))

        bb = np.bitwise_or(bb_moves, bb_takes)

        return self.get_moves_by_bb(src_index, bb)

    def get_moves_pawns_black(self):
        # get moves by the white pawn bb and returns them as [Move]
        moves:List[Move] = []
        bit = np.uint8(1)
        index = np.uint8(0)
        bb = self.bitboards[Color.BLACK, Piece.PAWN]
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_pawns_black_by_square(index)
            index += bit
            bb = np.right_shift(bb, bit)
        return moves

    def get_moves_pawns_black_by_square(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        bb_moves = np.uint64(0)
        bb_takes = np.uint64(0)

        src_bb = np.left_shift(np.uint64(1), src_index)

        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.right_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_H_BB)))
        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.right_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_A_BB)))
        bb_takes = np.bitwise_and(bb_takes, self.combined[0])

        if bb_takes:
            return self.get_moves_by_bb(src_index, bb_takes)

        bb_moves = np.bitwise_or(bb_moves, np.right_shift(src_bb, np.uint8(8)))
        if (48 <= src_index < 56):
            bb_moves = np.bitwise_or(bb_moves, np.right_shift(src_bb, np.uint8(16)))
        bb_moves = np.bitwise_and(bb_moves, np.bitwise_not(self.combined[1]))

        bb = np.bitwise_or(bb_moves, bb_takes)

        return self.get_moves_by_bb(src_index, bb)

    def get_moves_knights(self):
        # get moves by the self.player_to_move knights bb and returns them as [Move]
        moves:List[Move] = []
        index = np.uint8(0)
        bit = np.uint8(1)

        bb = self.bitboards[self.player_to_move, Piece.KNIGHT]
        
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_knights_by_square(index)
            index += bit
            bb = np.right_shift(bb, bit)
        return moves

    def get_moves_knights_by_square(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        bb = np.bitwise_and(KNIGHT_BB[src_index], np.bitwise_not(self.combined[self.player_to_move]))
        return self.get_moves_by_bb(src_index, bb)

    def get_moves_kings(self):
        # get moves by the self.player_to_move kings bb and returns them as [Move]
        moves:List[Move] = []
        index = np.uint8(0)
        bit = np.uint8(1)

        bb = self.bitboards[self.player_to_move, Piece.KING]
        
        while bb:
            if np.bitwise_and(bb, bit):
                moves += self.get_moves_kings_by_square(index)
            index += bit
            bb = np.right_shift(bb, bit)
        return moves
    
    def get_moves_kings_by_square(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        bb = np.bitwise_and(KING_BB[src_index], np.bitwise_not(self.combined[self.player_to_move]))
        return self.get_moves_by_bb(src_index, bb)

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
        self.bitboards[Color.WHITE, Piece.PAWN]   = np.uint64(0b1111111100000000)
        self.bitboards[Color.WHITE, Piece.KNIGHT] = np.uint64(0b01000010)
        self.bitboards[Color.WHITE, Piece.BISHOP] = np.uint64(0b00100100)
        self.bitboards[Color.WHITE, Piece.ROOK]   = np.uint64(0b10000001)
        self.bitboards[Color.WHITE, Piece.QUEEN]  = np.uint64(0b00010000)
        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b00001000)
        # black pieces
        self.bitboards[Color.BLACK, Piece.PAWN]   = np.uint64(0b11111111000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KNIGHT] = np.uint64(0b0100001000000000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.BISHOP] = np.uint64(0b0010010000000000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.ROOK]   = np.uint64(0b1000000100000000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.QUEEN]  = np.uint64(0b0001000000000000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b0000100000000000000000000000000000000000000000000000000000000000)

    def init_board_test_1(self):
        # initializes a board with the following configuration:
        # .. .. .. .. .. .. bK bR
        # .. .. .. .. .. .. bK bK
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # wK wK .. .. .. .. .. ..
        # wR wK .. .. .. .. .. ..

        # white pieces
        self.bitboards[Color.WHITE, Piece.ROOK] = np.uint64(0b10000000)
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b1100000001000000)
        # black pieces
        self.bitboards[Color.BLACK, Piece.ROOK] = np.uint64(0b0000000100000000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b0000001000000011000000000000000000000000000000000000000000000000)
    def init_board_test_2(self):
        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. bR .. .. .. ..
        # .. .. bK bK bK .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. wK wK wK .. .. ..
        # .. .. .. wR .. .. .. ..
        # .. .. .. .. .. .. .. ..

        # white pieces
        self.bitboards[Color.WHITE, Piece.ROOK] = np.uint64(0b0001000000000000)
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b001110000000000000000000)
        # black pieces
        self.bitboards[Color.BLACK, Piece.ROOK] = np.uint64(0b00010000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b001110000000000000000000000000000000000000000000)
    def init_board_test_3(self):
        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. bR .. .. .. ..
        # .. .. bK bK bK .. .. ..
        # .. .. wK wK wK .. .. ..
        # .. .. .. wR .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..

        # white pieces
        self.bitboards[Color.WHITE, Piece.ROOK] = np.uint64(0b000100000000000000000000)
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b00111000000000000000000000000000)
        # black pieces
        self.bitboards[Color.BLACK, Piece.ROOK] = np.uint64(0b000100000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b0011100000000000000000000000000000000000)
    def init_board_test_4(self):
        # initializes a board with the following configuration:
        # bK .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. wR

        # white pieces
        self.bitboards[Color.WHITE, Piece.ROOK] = np.uint64(0b1)
        # black pieces
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b1000000000000000000000000000000000000000000000000000000000000000)
    def init_board_test_5(self):
        # initializes a board with the following configuration:
        # bK bK bK .. .. .. .. ..
        # bP bP bP .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. wP wP wP
        # .. .. .. .. .. wK wK wK

        # white pieces
        self.bitboards[Color.WHITE, Piece.PAWN] = np.uint64(0b0000011100000000)
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b00000111)
        # black pieces
        self.bitboards[Color.BLACK, Piece.PAWN] = np.uint64(0b11100000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b1110000000000000000000000000000000000000000000000000000000000000)
    def init_board_test_6(self):
        # initializes a board with the following configuration:
        # bK bK bK bk .. .. .. ..
        # bP bP bP bk .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. wk wP wP wP
        # .. .. .. .. wk wK wK wK

        # white pieces
        self.bitboards[Color.WHITE, Piece.PAWN]   = np.uint64(0b0000011100000000)
        self.bitboards[Color.WHITE, Piece.KNIGHT] = np.uint64(0b0000100000001000)
        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b00000111)
        # black pieces
        self.bitboards[Color.BLACK, Piece.PAWN]   = np.uint64(0b11100000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KNIGHT] = np.uint64(0b0001000000010000000000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b1110000000000000000000000000000000000000000000000000000000000000)

def print_bb(bb:np.uint64):
    mask_bb = np.uint64(pow(2, 63))
    for i in range(64):
        if not(i%8):
            print()
        if np.bitwise_and(bb, mask_bb):
            print("1 ", end="")
        else:
            print(". ", end="")
        mask_bb = np.right_shift(mask_bb, np.uint8(1))
    print()

def main():
    chessboard = Chessboard()
    chessboard.init_board_standard()
    chessboard.translate_board()