import numpy as np
from enum import IntEnum
from typing import List
from typing import Tuple
from itertools import chain
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
    is_take        : bool

    def __init__(self, src_index:np.uint8, dst_index:np.uint8, promotion_type=None, is_take=False):
        self.src_index      = src_index
        self.dst_index      = dst_index
        self.promotion_type = promotion_type
        self.is_take        = is_take

    def __str__(self):
        # str function for moves
        return "src: " + str(self.src_index) + ", dst: " + str(self.dst_index) + ", pro: " + str(self.promotion_type) + ", take: " + str(self.is_take)

# function to calculate what type of move it is based on the source and destination indexes
# returns a value from 0 to 72 which is in the form of the output representation for the NN model
def calc_move(source: int, destination: int, promotion_piece: Piece):
    # board size

    src_col = source % 8
    src_row = source // 8

    dst_col = destination % 8
    dst_row = destination // 8

    # get the differences between the source and destination
    diff_col = src_col - dst_col
    diff_row = src_row - dst_row

    # iterate through all
    # and check if they match the given src/dst
    # list of tuples in form (col, row)
    # list is all viable moves for the output representation
    # knight moves in clockwise rotation starting from north

    # creating a list of test moves which will be iterated through
    # the first 56 moves are queen like moves in a clockwise pattern
    # the final 8 moves are knight like moves in a clockwise pattern
    # list(chain(*)) is used to flatten the list of lists to one list
    # the queen like moves are generated with list comprehension to increase readability
    tests = list(chain(*[
        # queen like moves
        [(0, a) for a in range(1, 8)],
        [(a, a) for a in range(1, 8)],
        [(a, 0) for a in range(1, 8)],
        [(a, -a) for a in range(1, 8)],
        [(0, -a) for a in range(1, 8)],
        [(-a, -a) for a in range(1, 8)],
        [(-a, 0) for a in range(1, 8)],
        [(-a, a) for a in range(1, 8)],

        # knight moves clockwise
        [(1, 2), (2, 1),
        (2, -1), (1, -2),
        (-1, -2), (-2, -1),
        (-2, 1), (-1, 2)]
                        ]))

    # testing for the 12 kinds of under promotions
    # in the following order:
    # go left + knight, go left + bishop, go left + rook, go left + king //
    # go up + knight, go up + bishop, go up + rook, go up + king //
    # go right + knight, go right + bishop, go right + rook, go right + king

    if promotion_piece is not None:

        if diff_col == -1: # left
            if promotion_piece == 1: # knight
                return 64
            if promotion_piece == 2: # bishop
                return 65
            if promotion_piece == 3: # rook
                return 66
            if promotion_piece == 5: # king
                return 67
        elif diff_col == 0:
            if promotion_piece == 1: # knight
                return 68
            if promotion_piece == 2: # bishop
                return 69
            if promotion_piece == 3: # rook
                return 70
            if promotion_piece == 5: # king
                return 71
        elif diff_col == -1:
            if promotion_piece == 1: # knight
                return 72
            if promotion_piece == 2: # bishop
                return 73
            if promotion_piece == 3: # rook
                return 74
            if promotion_piece == 5: # king
                return 75

    # for the first 64 available move types that aren't underpromotions
    for i in range(0,64):
        (t_col, t_row) = tests[i]
        if diff_col == t_col and diff_row == t_row:
            return i


class Chessboard():

    # main bitboard variable for representing the 12 bitboards.
    # The first index represents color, 0 = white, 1 = black
    # The second index represents piece-type.
    # 0 = pawns, 1 = knights, 2 = bishops, 3 = rooks, 4 = queens, 5 = kings
    bitboards : np.ndarray
    # index 0 = bitboard of combined white pieces
    # index 1 = bitboard of combined black pieces
    # index 2 = bitboard of combined black and white pieces
    combined  : np.ndarray
    # other logic
    player_to_move : Color
    not_player_to_move : Color

    # init empty board
    def __init__(self):
        self.bitboards = np.zeros((2, 6), dtype=np.uint64)
        self.combined  = np.zeros(3, dtype=np.uint64)
        self.player_to_move = Color.WHITE
        self.not_player_to_move = Color.BLACK

    def __str__(self):
        str_builder = []
        mask_bb = np.uint64(pow(2, 63))
        for i in range(64):
            if not i % 8:
                str_builder.append('\n')
            # if white pawn
            if np.bitwise_and(self.bitboards[0][0], mask_bb):
                str_builder.append('P ')
            # if black pawn
            elif np.bitwise_and(self.bitboards[1][0], mask_bb):
                str_builder.append('p ')

            # if white knight
            elif np.bitwise_and(self.bitboards[0][1], mask_bb):
                str_builder.append('K ')
            # if black knight
            elif np.bitwise_and(self.bitboards[1][1], mask_bb):
                str_builder.append('k ')

            # if white bishop
            elif np.bitwise_and(self.bitboards[0][2], mask_bb):
                str_builder.append('B ')
            # if black bishop
            elif np.bitwise_and(self.bitboards[1][2], mask_bb):
                str_builder.append('b ')

            # if white rook
            elif np.bitwise_and(self.bitboards[0][3], mask_bb):
                str_builder.append('R ')
            # if black rook
            elif np.bitwise_and(self.bitboards[1][3], mask_bb):
                str_builder.append('r ')

            # if white queen
            elif np.bitwise_and(self.bitboards[0][4], mask_bb):
                str_builder.append('Q ')
            # if black queen
            elif np.bitwise_and(self.bitboards[1][4], mask_bb):
                str_builder.append('q ')

            # if white king
            elif np.bitwise_and(self.bitboards[0][5], mask_bb):
                str_builder.append('K ')
            # if black king
            elif np.bitwise_and(self.bitboards[1][5], mask_bb):
                str_builder.append('k ')
            else:
                str_builder.append('. ')
            mask_bb = np.right_shift(mask_bb, np.uint8(1))

        return ''.join(str_builder)


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
        tmp = self.player_to_move
        self.player_to_move = self.not_player_to_move
        self.not_player_to_move = tmp

    def combine_bb(self):
        # combines bb of both player colors seperately and together
        index = 0
        for player in self.bitboards:
            combined = np.uint64(0)
            for bb in player:
                combined = np.bitwise_or(combined, bb)
            self.combined[index] = combined
            index += 1

        self.combined[2] = np.bitwise_or(self.combined[Color.WHITE], self.combined[Color.BLACK])
    def get_moves(self):
        # get legal moves given current state

        # combines the piece_type bitboards of both colors seperately, used in later calculations
        self.combine_bb()
        # vars to hold legal moves and takes
        moves:List[Move] = []
        takes:List[Move] = []
        # get takes by different piece types
        moves += self._get_moves()

        for m in moves:
            #print(str(m))
            if m.is_take:
                takes.append(m)

        #for piece_type in self.bitboards[self.player_to_move, :]:
            #print_bb(piece_type)

        #print(len(takes))
        #print(len(moves))

        if len(takes):
            return takes

        return moves

    def get_moves_by_bb(self, src_index:np.uint8, dst_bb:np.uint64, takes:bool=False):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = np.uint8(0)
        bit = np.uint8(1)
        while dst_bb:
            if np.bitwise_and(dst_bb, bit):
                moves.append(Move(src_index, dst_index, is_take=takes))
            dst_index += bit
            dst_bb = np.right_shift(dst_bb, bit)
        return moves
    
    def get_moves_by_bb_pawn(self, src_index:np.uint8, dst_bb:np.uint64, takes:bool=False):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = np.uint8(0)
        bit = np.uint8(1)
        while dst_bb:
            if np.bitwise_and(dst_bb, bit):
                if 8 > dst_index >= 56:
                    for piece_type in Piece:
                        moves.append(Move(src_index, dst_index, piece_type, takes))
                else:
                    moves.append(Move(src_index, dst_index, is_take=takes))
            dst_index += bit
            dst_bb = np.right_shift(dst_bb, bit)
        return moves
    
    def _get_moves(self):
        # get moves by the self.player_to_move kings bb and returns them as [Move]
        moves:List[Move] = []
        bit = np.uint8(1)
        for piece_type in Piece:
            src_index = np.uint(0)
            src_bb = self.bitboards[self.player_to_move, piece_type]
            while src_bb:
                if np.bitwise_and(src_bb, bit):
                    moves += self._get_moves_by_piece_type(piece_type, src_index)
                src_index += bit
                src_bb = np.right_shift(src_bb, bit)
        return moves
    
    def _get_moves_by_piece_type(self, piece_type:Piece, src_index:np.uint8):
        if piece_type == Piece.PAWN:
            if self.player_to_move == Color.WHITE:
                return self._get_moves_pawn_white(src_index)
            elif self.player_to_move == Color.BLACK:
                return self._get_moves_pawn_black(src_index)
            else:
                raise RuntimeError("Invalid player_to_move: %s" % str(self.player_to_move))
        elif piece_type == Piece.KNIGHT:
            return self._get_moves_knight(src_index)
        elif piece_type == Piece.BISHOP:
            return []
            #return self._get_moves_bishop(src_index)
        elif piece_type == Piece.ROOK:
            return []
            #return self._get_moves_rook(src_index)
        elif piece_type == Piece.QUEEN:
            return []
            #return self._get_moves_queen(src_index)
        elif piece_type == Piece.KING:
            return self._get_moves_king(src_index)
        else:
            raise RuntimeError("Invalid piece_type: %s" % str(piece_type))
        
    def _get_moves_pawn_white(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        src_bb = np.left_shift(np.uint64(1), src_index)

        bb_takes = np.bitwise_and(np.left_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_H_BB))
        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.left_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_A_BB)))
        bb_takes = np.bitwise_and(bb_takes, self.combined[Color.BLACK])

        if bb_takes: # if takes avaiable, no need to find non-takes
            return self.get_moves_by_bb_pawn(src_index, bb_takes, takes=True)

        bb_moves = np.left_shift(src_bb, np.uint8(8))
        if (8 <= src_index < 16):
            bb_moves = np.bitwise_or(bb_moves, np.left_shift(src_bb, np.uint8(16)))
        bb_moves = np.bitwise_and(bb_moves, np.bitwise_not(np.bitwise_or(self.combined[2], np.left_shift(np.bitwise_and(np.bitwise_not(src_bb), self.combined[2]), np.uint8(8)))))

        bb = np.bitwise_or(bb_moves, bb_takes)

        return self.get_moves_by_bb_pawn(src_index, bb)

    def _get_moves_pawn_black(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        src_bb = np.left_shift(np.uint64(1), src_index)

        bb_takes = np.bitwise_and(np.right_shift(src_bb, np.uint8(7)), np.bitwise_not(RANK_H_BB))
        bb_takes = np.bitwise_or(bb_takes, np.bitwise_and(np.right_shift(src_bb, np.uint8(9)), np.bitwise_not(RANK_A_BB)))
        bb_takes = np.bitwise_and(bb_takes, self.combined[Color.WHITE])

        if bb_takes: # if takes avaiable, no need to find non-takes
            return self.get_moves_by_bb_pawn(src_index, bb_takes, takes=True)

        bb_moves = np.right_shift(src_bb, np.uint8(8))
        if (48 <= src_index < 56):
            bb_moves = np.bitwise_or(bb_moves, np.right_shift(src_bb, np.uint8(16)))
        bb_moves = np.bitwise_and(bb_moves, np.bitwise_not(np.bitwise_or(self.combined[2], np.right_shift(np.bitwise_and(np.bitwise_not(src_bb), self.combined[2]), np.uint8(8)))))

        bb = np.bitwise_or(bb_moves, bb_takes)

        return self.get_moves_by_bb_pawn(src_index, bb)
    
    def _get_moves_knight(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        moves:List[Move] = []
        dst_bb = KNIGHT_BB[src_index]
        takes_bb = np.bitwise_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = np.bitwise_and(dst_bb, np.bitwise_not(self.combined[2]))
        moves += self.get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self.get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_king(self, src_index:np.uint8):
        # given a src_index, generate moves and returns them as [Move]
        moves:List[Move] = []
        dst_bb = KING_BB[src_index]
        takes_bb = np.bitwise_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = np.bitwise_and(dst_bb, np.bitwise_not(self.combined[2]))
        moves += self.get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self.get_moves_by_bb(src_index, moves_bb)
        return moves

    # translates the bitboard move representation into the output representation for the neural network
    # returns the output as an array of shape (1,1,8,8,73)
    def translate_moves_to_output(self):

        output = [[
            [[[0 for i in range(73)] for i in range(8)] for i in range(8)]
        ]]
        # fetch all the available moves
        moves = self.get_moves()
        # for every move, calculate what type value it has and set
        # the array index as 1 for the given move
        for move in moves:
            src_col = move.src_index % 8
            src_row = move.src_index // 8
            type_value = calc_move(move.src_index, move.dst_index, move.promotion_type)
            output[0][0][src_row][src_col][type_value] = 1 # PLACEHOLDER 1
            # TODO introduce some way to weigh each move according to the training data generated by the mcts
        # return all the moves in output representation
        return output

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

        #for row in range(0,8):
            #print(representation[0][row])
        return np.array(representation)

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
    def init_board_test_pawn_white_takes(self):
        # used for unit testing
        # move cardinality should be 21

        # initializes a board with the following configuration:
        # bK bK .. .. .. .. bK bK
        # wP wP .. wP .. .. wP wP
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. bK .. .. ..
        # .. .. .. wP .. .. wP ..
        # .. .. .. .. .. .. .. .. 

        self.bitboards[Color.WHITE, Piece.PAWN] = np.uint64(0b11010011000000000000000000000000000000000001001000000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b1100001100000000000000000000000000000000000010000000000000000000)
        self.player_to_move = Color.WHITE
    def init_board_test_pawn_white_moves(self):
        # used for unit testing
        # move cardinality should be 9

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. wP ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # bK .. .. .. .. .. .. ..
        # wP bK wP .. .. .. .. wP
        # .. .. .. .. .. .. .. .. 

        self.bitboards[Color.WHITE, Piece.PAWN] = np.uint64(0b00000010000000000000000000000000000000001010000100000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b100000000100000000000000)
        self.player_to_move = Color.WHITE
    def init_board_test_knight_white_takes(self):
        # used for unit testing
        # move cardinality should be 9

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. wk .. bK
        # wk wk wk .. wk .. .. ..
        # .. .. .. wk .. .. bK ..
        # .. bK .. .. wk .. .. ..
        # .. .. .. wk .. wk .. wk 

        self.bitboards[Color.WHITE, Piece.KNIGHT] = np.uint64(0b0000010011101000000100000000100000010101)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b0000000100000000000000100100000000000000)
        self.player_to_move = Color.WHITE
    def init_board_test_knight_white_moves(self):
        # used for unit testing
        # move cardinality should be 12

        # initializes a board with the following configuration:
        # wk .. .. .. .. .. .. ..
        # .. wk .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # bK .. .. .. .. .. .. ..
        # .. .. .. .. .. .. wk ..
        # .. .. .. .. .. .. .. wk 

        self.bitboards[Color.WHITE, Piece.KNIGHT] = np.uint64(0b1000000001000000000000000000000000000000000000000000001000000001)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b100000000000000000000000)
        self.player_to_move = Color.WHITE
    def init_board_test_bishop_white_takes(self):
        # used for unit testing
        # move cardinality should be 3

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. bK .. .. bK
        # .. .. .. bK .. .. wB ..
        # .. .. .. .. .. wK .. ..
        # .. wB .. .. .. .. .. ..
        # .. .. bK bK .. .. .. wB 
        self.bitboards[Color.WHITE, Piece.BISHOP] = np.uint64(0b0100000000000001)
        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b000001000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b0000100100010000000000000000000000110000)
        self.player_to_move = Color.WHITE
    def init_board_test_bishop_white_moves(self):
        # used for unit testing
        # move cardinality should be 12

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. bK .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. wB .. ..
        # .. .. .. .. .. wP .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. wB 

        self.bitboards[Color.WHITE, Piece.PAWN]   = np.uint64(0b000001000000000000000000)
        self.bitboards[Color.WHITE, Piece.BISHOP] = np.uint64(0b00000100000000000000000000000001)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b001000000000000000000000000000000000000000000000)
        self.player_to_move = Color.WHITE
    def init_board_test_rook_white_takes(self):
        # used for unit testing
        # move cardinality should be 3

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. bK
        # .. .. wR .. .. .. .. bK
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. bK .. wK .. .. wR 

        self.bitboards[Color.WHITE, Piece.ROOK] = np.uint64(0b0010000000000000000000000000000000000001)
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b00001000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b000000010000000100000000000000000000000000100000)
        self.player_to_move = Color.WHITE
    def init_board_test_rook_white_moves(self):
        # used for unit testing
        # move cardinality should be 29

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. wR
        # .. .. .. .. wR .. .. ..
        # .. .. bK .. wP .. .. wR 
        
        self.bitboards[Color.WHITE, Piece.PAWN] = np.uint64(0b00001000)
        self.bitboards[Color.WHITE, Piece.ROOK] = np.uint64(0b000000010000100000000001)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b00100000)
        self.player_to_move = Color.WHITE
    def init_board_test_queen_white_takes(self):
        # used for unit testing
        # move cardinality should be 4

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. wQ .. .. .. .. .. ..
        # .. wK .. .. bK .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. bK bK .. wQ .. bK ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. bK .. .. ..
        # .. .. .. .. .. .. .. .. 

        self.bitboards[Color.WHITE, Piece.QUEEN] = np.uint64(0b01000000000000000000000000001000000000000000000000000000)
        self.bitboards[Color.WHITE, Piece.KING]  = np.uint64(0b010000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING]  = np.uint64(0b000010000000000001100010000000000000100000000000)
        self.player_to_move = Color.WHITE
    def init_board_test_queen_white_moves(self):
        # used for unit testing
        # move cardinality should be 31

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. bK .. .. .. ..
        # .. .. .. .. .. wQ .. ..
        # .. .. .. .. .. wP .. ..
        # .. .. .. .. .. wP .. ..
        # .. .. .. bK .. wP .. wQ 

        self.bitboards[Color.WHITE, Piece.PAWN]  = np.uint64(0b000001000000010000000100)
        self.bitboards[Color.WHITE, Piece.QUEEN] = np.uint64(0b00000100000000000000000000000001)
        self.bitboards[Color.BLACK, Piece.KING]  = np.uint64(0b00010000)
        self.player_to_move = Color.WHITE
    def init_board_test_king_white_takes(self):
        # used for unit testing
        # move cardinality should be 8

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # bK .. .. .. .. .. .. ..
        # wK .. .. .. .. bK .. bK
        # .. .. .. .. .. bK wK bK
        # wK .. .. .. .. bK bK bK

        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b100000000000001010000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b10000000000001010000010100000111)
        self.player_to_move = Color.WHITE
    def init_board_test_king_white_moves(self):
        # used for unit testing
        # move cardinality should be 14

        # initializes a board with the following configuration:
        # wK .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. wK .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # bK .. .. .. .. .. .. wK 

        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b1000000000000000000000000000000000010000000000000000000000000001)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b10000000)
        self.player_to_move = Color.WHITE
    def init_board_test_pawn_black_takes(self):
        # used for unit testing
        # move cardinality should be 21

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. bP .. .. bP ..
        # .. .. .. .. wK .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # bP bP .. bP .. .. bP bP
        # wK wK .. .. .. .. wK wK 

        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b000010000000000000000000000000000000000011000011)
        self.bitboards[Color.BLACK, Piece.PAWN] = np.uint64(0b00010010000000000000000000000000000000001101001100000000)
        self.player_to_move = Color.BLACK
    def init_board_test_pawn_black_moves(self):
        # used for unit testing
        # move cardinality should be 9

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # bP wK bP .. .. .. .. bP
        # wK .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. bP ..
        # .. .. .. .. .. .. .. .. 

        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b01000000100000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.PAWN] = np.uint64(0b10100001000000000000000000000000000000000000001000000000)
        self.player_to_move = Color.BLACK
    def init_board_test_knight_black_takes(self):
        # used for unit testing
        # move cardinality should be 9

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. bk .. wK
        # bk bk bk .. bk .. .. ..
        # .. .. .. bk .. .. wK ..
        # .. wK .. .. bk .. .. ..
        # .. .. .. bk .. bk .. bk 

        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b0000000100000000000000100100000000000000)
        self.bitboards[Color.BLACK, Piece.KNIGHT] = np.uint64(0b0000010011101000000100000000100000010101)
        self.player_to_move = Color.BLACK
    def init_board_test_knight_black_moves(self):
        # used for unit testing
        # move cardinality should be 12

        # initializes a board with the following configuration:
        # bk .. .. .. .. .. .. ..
        # .. bk .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # wK .. .. .. .. .. .. ..
        # .. .. .. .. .. .. bk ..
        # .. .. .. .. .. .. .. bk 

        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b100000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KNIGHT] = np.uint64(0b1000000001000000000000000000000000000000000000000000001000000001)
        self.player_to_move = Color.BLACK
    def init_board_test_bishop_black_takes(self):
        # used for unit testing
        # move cardinality should be 3

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. wK .. .. wK
        # .. .. .. wK .. .. bB ..
        # .. .. .. .. .. bK .. ..
        # .. bB .. .. .. .. .. ..
        # .. .. wK wK .. .. .. bB 

        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b0000100100010000000000000000000000110000)
        self.bitboards[Color.BLACK, Piece.BISHOP] = np.uint64(0b0100000000000001)
        self.bitboards[Color.BLACK, Piece.KING]   = np.uint64(0b000001000000000000000000)
        self.player_to_move = Color.BLACK
    def init_board_test_bishop_black_moves(self):
        # used for unit testing
        # move cardinality should be 10

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. wK .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. bP .. ..
        # .. .. .. .. .. bB .. ..
        # .. .. .. .. .. .. .. bB 

        self.bitboards[Color.WHITE, Piece.KING]   = np.uint64(0b001000000000000000000000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.PAWN]   = np.uint64(0b000001000000000000000000)
        self.bitboards[Color.BLACK, Piece.BISHOP] = np.uint64(0b0000010000000001)
        self.player_to_move = Color.BLACK
    def init_board_test_rook_black_takes(self):
        # used for unit testing
        # move cardinality should be 3

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. wK
        # .. .. bR .. .. .. .. wK
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. wK .. bK .. .. bR 

        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b000000010000000100000000000000000000000000100000)
        self.bitboards[Color.BLACK, Piece.ROOK] = np.uint64(0b0010000000000000000000000000000000000001)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b00001000)
        self.player_to_move = Color.BLACK
    def init_board_test_rook_black_moves(self):
        # used for unit testing
        # move cardinality should be 29

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. bR
        # .. .. .. .. bR .. .. ..
        # .. .. wK .. bP .. .. bR 
        
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b00100000)
        self.bitboards[Color.BLACK, Piece.PAWN] = np.uint64(0b00001000)
        self.bitboards[Color.BLACK, Piece.ROOK] = np.uint64(0b000000010000100000000001)
        self.player_to_move = Color.BLACK
    def init_board_test_queen_black_takes(self):
        # used for unit testing
        # move cardinality should be 4

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. bQ .. .. .. .. .. ..
        # .. bK .. .. wK .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. wK wK .. bQ .. wK ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. wK .. .. ..
        # .. .. .. .. .. .. .. .. 

        self.bitboards[Color.WHITE, Piece.KING]  = np.uint64(0b000010000000000001100010000000000000100000000000)
        self.bitboards[Color.BLACK, Piece.QUEEN] = np.uint64(0b01000000000000000000000000001000000000000000000000000000)
        self.bitboards[Color.BLACK, Piece.KING]  = np.uint64(0b010000000000000000000000000000000000000000000000)
        self.player_to_move = Color.BLACK
    def init_board_test_queen_black_moves(self):
        # used for unit testing
        # move cardinality should be 31

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. wK .. .. .. ..
        # .. .. .. .. .. bQ .. ..
        # .. .. .. .. .. bP .. ..
        # .. .. .. .. .. bP .. ..
        # .. .. .. wK .. bP .. bQ 

        self.bitboards[Color.WHITE, Piece.KING]  = np.uint64(0b00010000)
        self.bitboards[Color.BLACK, Piece.PAWN]  = np.uint64(0b000001000000010000000100)
        self.bitboards[Color.BLACK, Piece.QUEEN] = np.uint64(0b00000100000000000000000000000001)
        self.player_to_move = Color.BLACK
    def init_board_test_king_black_takes(self):
        # used for unit testing
        # move cardinality should be 8

        # initializes a board with the following configuration:
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # wK .. .. .. .. .. .. ..
        # bK .. .. .. .. wK .. wK
        # .. .. .. .. .. wK bK wK
        # bK .. .. .. .. wK wK wK

        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b10000000000001010000010100000111)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b100000000000001010000000)
        self.player_to_move = Color.BLACK
    def init_board_test_king_black_moves(self):
        # used for unit testing
        # move cardinality should be 14

        # initializes a board with the following configuration:
        # bK .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. bK .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # .. .. .. .. .. .. .. ..
        # wK .. .. .. .. .. .. bK 
        
        self.bitboards[Color.WHITE, Piece.KING] = np.uint64(0b10000000)
        self.bitboards[Color.BLACK, Piece.KING] = np.uint64(0b1000000000000000000000000000000000010000000000000000000000000001)
        self.player_to_move = Color.BLACK
    def init_board_test_enpassante_white(self):
        pass
    def init_board_test_enpassante_black(self):
        pass

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
    chessboard.translate_moves_to_output()

if __name__ == '__main__':
    main()
