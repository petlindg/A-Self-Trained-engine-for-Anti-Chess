import numpy as np
from numpy import uint64 as u64
from numpy import uint8 as u8
from numpy import left_shift as ls
from numpy import right_shift as rs
from numpy import bitwise_or as b_or
from numpy import bitwise_and as b_and
from numpy import bitwise_not as b_not
from numpy import bitwise_xor as b_xor
from numpy import zeros, ndarray, uint, array

from enum import IntEnum
from typing import List
from itertools import chain

# -----------------------------
# ----------- TO DO -----------
# -----------------------------

class LOOKUP_TABLES():
    """
    A class to store lookuptables for generating moves and masking parts of a bitboard
    """

    def __init__(self):
        # ranks by number
        self.RANK_8_BB = u64(0b1111111100000000000000000000000000000000000000000000000000000000)
        self.RANK_7_BB = u64(0b11111111000000000000000000000000000000000000000000000000)
        self.RANK_6_BB = u64(0b111111110000000000000000000000000000000000000000)
        self.RANK_5_BB = u64(0b1111111100000000000000000000000000000000)
        self.RANK_4_BB = u64(0b11111111000000000000000000000000)
        self.RANK_3_BB = u64(0b111111110000000000000000)
        self.RANK_2_BB = u64(0b1111111100000000)
        self.RANK_1_BB = u64(0b11111111)

        self.rank_nr_list: list[u64] = [self.RANK_8_BB, self.RANK_7_BB, self.RANK_6_BB, self.RANK_5_BB, self.RANK_4_BB, self.RANK_3_BB, self.RANK_2_BB, self.RANK_1_BB]

        # ranks by letter
        self.FILE_A_BB = u64(0b1000000010000000100000001000000010000000100000001000000010000000)
        self.FILE_B_BB = u64(0b0100000001000000010000000100000001000000010000000100000001000000)
        self.FILE_C_BB = u64(0b0010000000100000001000000010000000100000001000000010000000100000)
        self.FILE_D_BB = u64(0b0001000000010000000100000001000000010000000100000001000000010000)
        self.FILE_E_BB = u64(0b0000100000001000000010000000100000001000000010000000100000001000)
        self.FILE_F_BB = u64(0b0000010000000100000001000000010000000100000001000000010000000100)
        self.FILE_G_BB = u64(0b0000001000000010000000100000001000000010000000100000001000000010)
        self.FILE_H_BB = u64(0b0000000100000001000000010000000100000001000000010000000100000001)

        self.rank_l_list: list[u64] = [self.FILE_A_BB, self.FILE_B_BB, self.FILE_C_BB, self.FILE_D_BB, self.FILE_E_BB, self.FILE_F_BB, self.FILE_G_BB, self.FILE_H_BB]

        self.KNIGHT_BB = zeros(64, dtype=u64)
        self.KING_BB = zeros(64, dtype=u64)

        self.FIRST_RANK_ATTACKS = self._rank_masks_init()
        self.FILE_H_ATTACKS = self._file_masks_init()
        self.DIAG_MASKS = self._diag_masks_init()
        self.ANTIDIAG_MASKS = self._antidiag_masks_init()
        self.FIRST_RANK_ATTACKS = self._first_rank_attacks_init()
        self.FILE_H_ATTACKS = self._file_h_attacks_init()
        self.KNIGHT_BB = self._knight_bb_init()
        self.KING_BB = self._king_bb_init()

    def _rank_masks_init(self):
        arr = zeros(64, dtype=u64)
        for index in range(64):
            if not index%8:
                rank = ls(self.RANK_1_BB, u8(index))
            arr[index] = rank
        return arr
    def _file_masks_init(self):
        arr = zeros(64, dtype=u64)
        for index in range(64):
            if not index%8:
                file = self.FILE_H_BB
            else:
                file = ls(file, u8(1))
            arr[index] = file
        return arr
    def _diag_masks_init(self):
        arr = zeros(64, dtype=u64)
        bit = u64(1)
        for index in range(64):
            bb = u64(0)
            sq = ls(bit, u64(index))
            while(sq):
                bb = b_or(bb, sq)
                sq = b_and(ls(sq, uint(7)), b_not(self.FILE_A_BB))
            sq = ls(bit, u64(index))
            while(sq):
                bb = b_or(bb, sq)
                sq = b_and(rs(sq, uint(7)), b_not(self.FILE_H_BB))
            arr[index] = bb
        return arr
    def _antidiag_masks_init(self):
        arr = zeros(64, dtype=u64)
        bit = u64(1)
        for index in range(64):
            bb = u64(0)
            sq = ls(bit, u64(index))
            while(sq):
                bb = b_or(bb, sq)
                sq = b_and(ls(sq, uint(9)), b_not(self.FILE_H_BB))
            sq = ls(bit, u64(index))
            while(sq):
                bb = b_or(bb, sq)
                sq = b_and(rs(sq, uint(9)), b_not(self.FILE_A_BB))
            arr[index] = bb
        return arr
    def _calc_first_rank_attacks(self, index:u8, occ:u8):
        attacks = u8(0)
        bit = u8(1)
        i = ls(bit, index+bit)
        while i:
            current_bit = b_and(i, occ)
            attacks = b_or(attacks, i)
            if current_bit:
                break
            i = ls(i, bit)
        i = ls(bit, index-bit)
        while i:
            current_bit = b_and(i, occ)
            attacks = b_or(attacks, i)
            if current_bit:
                break
            i = rs(i, bit)
        return attacks
    def _calc_file_h_attacks(self, index:u8, occ:u8):

        index = u64(index)
        occ = u64(occ)

        attacks = u64(0)
        bit = u64(1)
        byte = u64(8)
        i = ls(bit, index+bit)
        j = index+bit
        while i:
            current_bit = b_and(i, occ)
            attacks = b_or(attacks, ls(bit, j*byte))
            if current_bit:
                break
            i = ls(i, bit)
            j += bit
        i = ls(bit, index-bit)
        j = index-bit
        while i:
            current_bit = b_and(i, occ)
            attacks = b_or(attacks, ls(bit, j*byte))
            if current_bit:
                break
            i = rs(i, bit)
            j -= bit
        return attacks
    def _first_rank_attacks_init(self):
        arr = zeros((8, 256), dtype=u8)
        for index in range(8):
            for occ in range(256):
                arr[index, occ] = self._calc_first_rank_attacks(index, occ)
        return arr
    def _file_h_attacks_init(self):
        arr = zeros((8, 256), dtype=u64)
        for index in range(8):
            for occ in range(256):
                arr[index, occ] = self._calc_file_h_attacks(index, occ)
        return arr
    def _knight_bb_init(self):
        # Init of movegeneration bitboards for knights.
        # The bitboard of bbs[index] represents the bitboard with all possible destinationsquares given a source square = index
        bbs = zeros(64, dtype=u64)
        src_bb = u64(1) # source bb to generate moves from
        for index in range(u64(64)):
            dst_bb = u64(0) # destination bb to track all possible move destinations

            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(17)), b_not(self.FILE_A_BB)))
            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(15)), b_not(self.FILE_H_BB)))
            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(10)), b_not(b_or(self.FILE_A_BB, self.FILE_B_BB))))
            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(6)),  b_not(b_or(self.FILE_G_BB, self.FILE_H_BB))))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(6)),  b_not(b_or(self.FILE_A_BB, self.FILE_B_BB))))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(10)), b_not(b_or(self.FILE_G_BB, self.FILE_H_BB))))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(15)), b_not(self.FILE_A_BB)))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(17)), b_not(self.FILE_H_BB)))

            bbs[index] = dst_bb
            src_bb = ls(src_bb, u8(1)) # shift src_bb to match index
        return bbs
    def _king_bb_init(self):
        # Init of movegeneration bitboards for kings.
        # The bitboard of bbs[index] represents the bitboard with all possible destinationsquares given a source square = index
        bbs = zeros(64, dtype=u64)
        src_bb = u64(1) # source bb to generate moves from
        for index in range(u64(64)):
            dst_bb = u64(0) # destination bb to track all possible move destinations

            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(9)), b_not(self.FILE_A_BB)))
            dst_bb = b_or(dst_bb,                rs(src_bb, u8(8)))
            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(7)), b_not(self.FILE_H_BB)))
            dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(1)), b_not(self.FILE_A_BB)))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(1)),  b_not(self.FILE_H_BB)))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(7)),  b_not(self.FILE_A_BB)))
            dst_bb = b_or(dst_bb,                ls(src_bb, u8(8)))
            dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(9)),  b_not(self.FILE_H_BB)))

            bbs[index] = dst_bb
            src_bb = ls(src_bb, u8(1)) # shift src_bb to match index
        return bbs

LOOKUP = LOOKUP_TABLES()

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
    """
    Class for representing a move, used for functions within the Chessboard class
    """

    src_index      : u8
    dst_index      : u8
    promotion_type : Piece
    is_take        : bool

    def __init__(self, src_index:u8, dst_index:u8, promotion_type=None, is_take=False):
        self.src_index      = src_index
        self.dst_index      = dst_index
        self.promotion_type = promotion_type
        self.is_take        = is_take

    def __str__(self):
        # str function for moves
        return str(move_to_algebraic(self))
               #+ "src: " + str(self.src_index) + ", dst: " + str(self.dst_index) + ", pro: " + str(self.promotion_type) + ", take: " + str(self.is_take)
    def __eq__(self, move):
        return self.src_index == move.src_index and self.dst_index == move.dst_index and self.promotion_type == move.promotion_type


def move_to_algebraic(move):
    """Function for translating a move into a string in algebraic notation

    :param move: Move Class
    :return: String
    """

    cols = ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
    src = move.src_index
    src_row = str(int(src//8 + 1))
    src_col = cols[int(src%8)]

    dst = move.dst_index
    dst_row = str(int(dst//8 + 1))
    dst_col = cols[int(dst%8)]

    return src_col + src_row + dst_col + dst_row

def algebraic_to_move(s):
    """Takes an algebraic representation string s and returns the move representation for this string

    :param s: String
    :return: Class Move
    """
    s = list(s.lower())
    fst_c = ''
    fst_d = 0
    snd_c = ''
    snd_d = 0
    counter = 0
    for character in s:
        if counter == 4:
            break
        if counter == 0 and character.isalpha():
            fst_c = str(character)
        if counter == 1 and character.isdigit():
            fst_d = int(character)
        if counter == 2 and character.isalpha():
            snd_c = str(character)
        if counter == 3 and character.isdigit():
            snd_d = int(character)
        counter +=1
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # default values in the case that the input is wrong and the characters arent valid
    if fst_c not in cols:
        src_col = 1
    if snd_c not in cols:
        dst_col = 1
    if fst_d > 8:
        fst_d = 0
    if snd_d > 8:
        snd_d = 0

    # convert letters to integers
    for i, c in enumerate(cols):
        if fst_c == c:
            src_col = i + 1
        if snd_c == c:
            dst_col = i + 1

    src_index = np.uint8(-src_col + 8 * fst_d)
    dst_index = np.uint8(-dst_col + 8 * snd_d)
    return Move(src_index, dst_index)

def calc_move(source: int, destination: int, promotion_piece: Piece):
    # function to calculate what type of move it is based on the source and destination indexes
    # returns a value from 0 to 78 which is in the form of the output representation for the NN model
    # board size

    src_col = int(source % 8)
    src_row = int(source // 8)

    dst_col = int(destination % 8)
    dst_row = int(destination // 8)

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
            return 63 + promotion_piece
        elif diff_col == 0: # middle
            return 68 + promotion_piece
        elif diff_col == 1: # right
            return 73 + promotion_piece
            # max is 73+5=78, as index makes size 79 which matches output size


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
    bitboards : ndarray
    # index 0 = bitboard of combined white pieces
    # index 1 = bitboard of combined black pieces
    # index 2 = bitboard of combined black and white pieces
    combined : ndarray
    # bitboard to keep track of en-passant-able squares
    enpassante : u64
    # other logic
    player_to_move : Color
    not_player_to_move : Color
    no_progress_counter : List[u8]
    # a list of bitboards that represent the state
    repetitions_list : List[ndarray]
    # a bitboard to keep track of combined pawn positions
    pawns : u64
    # keeping track of number of pieces on the board
    piece_count : u8
    move_counter : u8
    

    # init empty board
    def __init__(self, fen:str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1"):
        """
        Initializes the chessboard. By default with standard chess configuration
        alternatively by a string given in FEN format, excluding the field for castling.

        :param fen: String stating boardstate in FEN format, excluding the castling rights field
        """
        self.bitboards = zeros((2, 6), dtype=u64)
        self.combined  = zeros(3, dtype=u64)
        self.repetitions_list = []
        self.no_progress_counter = []
        self._init_fen(fen)
        self.pawns = self._get_pawns()
        self.piece_count = self._get_piece_count()
        
    def __str__(self):
        str_builder = ['a b c d e f g h \n']
        mask_bb = u64(pow(2, 63))
        n = 8
        for i in range(64):
            if not i % 8:
                if i != 0:
                    str_builder.append(f'{n}\n')
                    n = n - 1
            # if white pawn
            if b_and(self.bitboards[0][0], mask_bb):
                str_builder.append('wP')
            # if black pawn
            elif b_and(self.bitboards[1][0], mask_bb):
                str_builder.append('bP')

            # if white knight
            elif b_and(self.bitboards[0][1], mask_bb):
                str_builder.append('wH')
            # if black knight
            elif b_and(self.bitboards[1][1], mask_bb):
                str_builder.append('bH')

            # if white bishop
            elif b_and(self.bitboards[0][2], mask_bb):
                str_builder.append('wB')
            # if black bishop
            elif b_and(self.bitboards[1][2], mask_bb):
                str_builder.append('bB')

            # if white rook
            elif b_and(self.bitboards[0][3], mask_bb):
                str_builder.append('wR')
            # if black rook
            elif b_and(self.bitboards[1][3], mask_bb):
                str_builder.append('bR')

            # if white queen
            elif b_and(self.bitboards[0][4], mask_bb):
                str_builder.append('wQ')
            # if black queen
            elif b_and(self.bitboards[1][4], mask_bb):
                str_builder.append('bQ')

            # if white king
            elif b_and(self.bitboards[0][5], mask_bb):
                str_builder.append('wK')
            # if black king
            elif b_and(self.bitboards[1][5], mask_bb):
                str_builder.append('bK')
            else:
                str_builder.append('. ')
            mask_bb = rs(mask_bb, u8(1))
        str_builder.append('1')

        return ''.join(str_builder)


    def get(self):
        """
        :return: shape (2,6) dtype=u64 ndarray where [i,j] represents the bitboard of Color i and Piece j
        """
        return self.bitboards

    def try_move(self, src_index:int, dst_index:int, promotion_type:Piece=None):
        """
        Given input parameters, makes a move on the board if it's legal

        :param src_index: Index of the source square of a move in range 0-63
        :param dst_iondex: Index of the destination square of a move in range 0-63
        :param promotion_type: Optional promotion type of a move reprenented as Piece (range 1-5)
        :return: True if move was legal, False otherwise
        """

        move = Move(u8(src_index), u8(dst_index), promotion_type) # create function parameters into a Move()
        moves = self.get_moves() # get legal moves
        for m in moves:
            # checks if input move is in legal moves
            if move.src_index == m.src_index and move.dst_index == m.dst_index and move.promotion_type != Piece.PAWN:
                self.move(move)
                return True
        return False

    def get_game_status(self):
        """
        Gets the status of the game in the current state

        :return: 0 if white wins, 1 if black wins, 2 if game is draw, 3 if game is ongoing
        """
    
        moves = self.get_moves()
        if not len(moves):
            return self.player_to_move
        if self._check_repetitions():
            return 2
        if self._check_no_progress():
            return 2
        return 3
    
    def get_player(self):
        """
        Gets current player

        :return: Color of current player to move
        """

        return self.player_to_move
    
    def is_draw(self):
        """
        Checks if the game is a draw or not based on repetitions and no-progress

        :return: True if game is draw, False otherwise
        """

        if self._check_repetitions():
            return True
        if self._check_no_progress():
            return True
        return False

    def _check_repetitions(self):
        if self._get_repetitions() == 2:
            return True
        return False
    
    def _check_no_progress(self):
        if self.no_progress_counter[-1] == 50:
            return True
        return False

    def move(self, move:Move):
        """
        Executes a move and updates all the parameters of the Chessboard

        :param move: Type Move, represents the move to be executed
        :return: True if game is drawed after move is executed, False otherwise
        """

        # update piececount for _update_no_progress()
        self.piece_count = self._get_piece_count()
        self.pawns = self._get_pawns()

        self._update_repetitions()
        self._update_bitboards(move)
        self._update_no_progress()
        self._update_player()
        self._update_move_counter()
        return self.is_draw()
    
    def _init_fen(self, fen:str):
        arr = fen.split()

        board = arr[0].split('/')
        numbers = "12345678"
        bit = u64(1)

        index = u64(63)
        for rank in board:
            for c in rank:
                if c in numbers:
                    index -= u64(c)
                else:
                    bb_indexes = self._char_to_bb_indexes(c)
                    self.bitboards[bb_indexes] = b_or(self.bitboards[bb_indexes], ls(bit, index))
                    index -= bit

        player = arr[1]
        self.player_to_move = self._char_to_player(player)
        if self.player_to_move == Color.WHITE:
            self.not_player_to_move = Color.BLACK
        else:
            self.not_player_to_move = Color.WHITE

        enpassante = arr[2]
        if enpassante == '-':
            self.enpassante = u64(0)
        else:
            self.enpassante = ls(bit, u8(self._alg_sq_to_index(enpassante)))

        no_progress = arr[3]
        self.no_progress_counter.append(u8(no_progress))

        move_counter = arr[4]
        self.move_counter = u8(move_counter)

    def _alg_sq_to_index(self, alg:str):
        file_char = alg[0]
        rank_char = alg[1]
        return 72-ord(file_char) + (int(rank_char)-1)*8
    def _char_to_player(self, c:str):
        if c=='w':
            return Color.WHITE
        elif c=='b':
            return Color.BLACK
        else:
            raise RuntimeError("Invalid character as player color, %s" % c)
        

    def _char_to_bb_indexes(self, c:str):
        if c == 'P':
            return (Color.WHITE, Piece.PAWN)
        elif c == 'N':
            return (Color.WHITE, Piece.KNIGHT)
        elif c == 'B':
            return (Color.WHITE, Piece.BISHOP)
        elif c == 'R':
            return (Color.WHITE, Piece.ROOK)
        elif c == 'Q':
            return (Color.WHITE, Piece.QUEEN)
        elif c == 'K':
            return (Color.WHITE, Piece.KING)
        elif c == 'p':
            return (Color.BLACK, Piece.PAWN)
        elif c == 'n':
            return (Color.BLACK, Piece.KNIGHT)
        elif c == 'b':
            return (Color.BLACK, Piece.BISHOP)
        elif c == 'r':
            return (Color.BLACK, Piece.ROOK)
        elif c == 'q':
            return (Color.BLACK, Piece.QUEEN)
        elif c == 'k':
            return (Color.BLACK, Piece.KING)
        else:
            raise RuntimeError("Invalid character as Color and Piece, %s" % c)

    def get_fen(self):
        """
        :return: The FEN representation of the state of the chessboard
        """
        bit = u64(1)

        fen = ""

        board = ""
        bb = ls(bit, u8(63))
        for rank in range(8):
            empty_counter = 0
            for file in range(8):
                occupied = False
                for player in Color:
                    for piece_type in Piece:
                        if b_and(self.bitboards[player, piece_type], bb):
                            if empty_counter:
                                board += str(empty_counter)
                                empty_counter = 0
                            board += self._color_piece_type_to_char(player, piece_type)
                            occupied = True
                if not occupied:
                    empty_counter += 1
                if file==7 and empty_counter:
                    board += str(empty_counter)
                bb = rs(bb, bit)
            if rank!=7:
                board += '/'
        fen += board + " "

        player = self._player_to_char(self.player_to_move)
        fen += player + " "

        enpassante_index = u8(0)
        enpassante_bb = self.enpassante
        if enpassante_bb:
            while enpassante_bb:
                if b_and(enpassante_bb, bit):
                    break
                enpassante_index += 1
                enpassante_bb = rs(enpassante_bb, bit)
            enpassante = self._index_to_alg_sq(enpassante_index)
        else:
            enpassante = '-'
        fen += enpassante + " "

        no_progress = str(self.no_progress_counter[-1])
        fen += no_progress + " "
        
        move_counter = str(self.move_counter)
        fen += move_counter

        return fen

    def _index_to_alg_sq(self, index:u8):
        rank = str(rs(index, u8(3))+1)
        file = b_and(index, u8(0b000_111))
        file = chr(65+(7-file))
        return file + rank

    def _player_to_char(self, color:Color):
        if color == Color.WHITE:
            return 'w'
        elif color == Color.BLACK:
            return 'b'
        else:
            raise RuntimeError("Invalid Color, %s" % str(color))

    def _color_piece_type_to_char(self, color:Color, piece_type:Piece):
        if color == Color.WHITE:
            if piece_type == Piece.PAWN:
                return 'P'
            elif piece_type == Piece.KNIGHT:
                return 'N'
            elif piece_type == Piece.BISHOP:
                return 'B'
            elif piece_type == Piece.ROOK:
                return 'R'
            elif piece_type == Piece.QUEEN:
                return 'Q'
            elif piece_type == Piece.KING:
                return 'K'
            else:
                raise RuntimeError("Invalid Piece, %s" % str(piece_type))
        elif color == Color.BLACK:
            if piece_type == Piece.PAWN:
                return 'p'
            elif piece_type == Piece.KNIGHT:
                return 'n'
            elif piece_type == Piece.BISHOP:
                return 'b'
            elif piece_type == Piece.ROOK:
                return 'r'
            elif piece_type == Piece.QUEEN:
                return 'q'
            elif piece_type == Piece.KING:
                return 'k'
            else:
                raise RuntimeError("Invalid Piece, %s" % str(piece_type))
        else:
            raise RuntimeError("Invalid Color, %s" % str(color))

    def _update_bitboards(self, move:Move):
        src_bb = ls(u64(1), move.src_index)
        dst_bb = ls(u64(1), move.dst_index)
        promotion_type = move.promotion_type

        enpassante = self.enpassante
        self.enpassante = u64(0)

        for piece_type in Piece:
            self.bitboards[self.not_player_to_move, piece_type] = b_and(self.bitboards[self.not_player_to_move, piece_type], b_not(dst_bb))
            if b_and(self.bitboards[self.player_to_move, piece_type], src_bb):
                if piece_type == Piece.PAWN:
                    self._move_pawn(src_bb, dst_bb, enpassante, promotion_type)
                else:
                    self._move_piece(src_bb, dst_bb, piece_type)

    def _move_pawn(self, src_bb:u64, dst_bb:u64, enpassante:u64, promotion_type:Piece):
        self.bitboards[self.player_to_move, Piece.PAWN] = b_xor(self.bitboards[self.player_to_move, Piece.PAWN], src_bb)
        if promotion_type:
            self._move_promote(dst_bb, promotion_type)
        elif b_and(dst_bb, enpassante):
            self._move_enpassante(dst_bb)
        else:
            self.bitboards[self.player_to_move, Piece.PAWN] = b_or(self.bitboards[self.player_to_move, Piece.PAWN], dst_bb)

    def _move_promote(self, dst_bb:u64, promotion_type:Piece):
        self.bitboards[self.player_to_move, promotion_type] = b_or(self.bitboards[self.player_to_move, promotion_type], dst_bb)

    def _move_enpassante(self, dst_bb:u64):
        self.bitboards[self.player_to_move, Piece.PAWN] = b_or(self.bitboards[self.player_to_move, Piece.PAWN], dst_bb)
        if self.player_to_move == Color.WHITE:
            self.bitboards[self.not_player_to_move, Piece.PAWN] = b_xor(self.bitboards[self.not_player_to_move, Piece.PAWN], rs(dst_bb, u64(8)))
        else:
            self.bitboards[self.not_player_to_move, Piece.PAWN] = b_xor(self.bitboards[self.not_player_to_move, Piece.PAWN], ls(dst_bb, u64(8)))

    def _move_piece(self, src_bb:u64, dst_bb:u64, piece_type:Piece):
        self.bitboards[self.player_to_move, piece_type] = b_xor(self.bitboards[self.player_to_move, piece_type], src_bb)
        self.bitboards[self.player_to_move, piece_type] = b_or(self.bitboards[self.player_to_move, piece_type], dst_bb)

    def _update_repetitions(self):
        self.repetitions_list.append(self.bitboards.copy())

    def _update_no_progress(self):
        if self._update_piece_count() or self._update_pawns():
            self.no_progress_counter.append(u8(0))
        else:
            self.no_progress_counter.append(self.no_progress_counter[-1]+u8(1))

    def _update_piece_count(self):
        new_piece_count = self._get_piece_count()
        if new_piece_count != self.piece_count:
            self.piece_count = new_piece_count
            return True
        return False
    
    def _update_pawns(self):
        new_pawns = self._get_pawns()
        if new_pawns != self.pawns:
            self.pawns = new_pawns
            return True
        return False

    def _get_pawns(self):
        return b_or(self.bitboards[Color.WHITE, Piece.PAWN], self.bitboards[Color.BLACK, Piece.PAWN])
    
    def unmove(self):
        self._update_player()
        self._reverse_move_counter()
        self.bitboards = self.repetitions_list.pop()
        self.no_progress_counter.pop()

    def _update_player(self):
        tmp = self.player_to_move
        self.player_to_move = self.not_player_to_move
        self.not_player_to_move = tmp

    def _update_move_counter(self):
        if self.player_to_move == Color.WHITE:
            self.move_counter += u8(1)

    def _reverse_move_counter(self):
        if self.player_to_move == Color.BLACK:
            self.move_counter -= u8(1)

    def _get_piece_count(self):
        bit = u8(1)
        count = 0
        for player in self.bitboards:
            for piece_bb in player:
                while piece_bb:
                    if b_and(piece_bb, bit):
                        count += 1
                    piece_bb = rs(piece_bb, bit)
        return count
    
    def _get_repetitions(self):
        repetitions = 0
        for state in self.repetitions_list:
            if self._bbs_equal(state, self.bitboards):
                repetitions += 1
        return repetitions

    def _bbs_equal(self, bbs1:ndarray, bbs2:ndarray):
        for player in Color:
            for piece_type in Piece:
                if bbs1[player, piece_type] != bbs2[player, piece_type]:
                    return False
        return True

    def _combine_bb(self):
        # combines bb of both player colors seperately and together
        index = 0
        for player in self.bitboards:
            combined = u64(0)
            for bb in player:
                combined = b_or(combined, bb)
            self.combined[index] = combined
            index += 1

        self.combined[2] = b_or(self.combined[Color.WHITE], self.combined[Color.BLACK])

    def get_moves(self):
        """
        Gets the legal moves of the current state

        :return: A List[Move] of legal moves
        """

        # combines the piece_type bitboards of both colors seperately, used in later calculations
        self._combine_bb()
        # vars to hold legal moves and takes
        moves:List[Move] = []
        takes:List[Move] = []
        # get takes by different piece types
        moves += self._get_moves()

        for m in moves:
            if m.is_take:
                takes.append(m)

        if len(takes):
            return takes

        return moves

    def _get_moves_by_bb(self, src_index:u8, dst_bb:u64, takes:bool=False):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = u8(0)
        bit = u8(1)
        while dst_bb:
            if b_and(dst_bb, bit):
                moves.append(Move(src_index, dst_index, is_take=takes))
            dst_index += bit
            dst_bb = rs(dst_bb, bit)
        return moves
    
    def _get_moves_by_bb_pawn(self, src_index:u8, dst_bb:u64, takes:bool=False):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = u8(0)
        bit = u8(1)
        while dst_bb:
            if b_and(dst_bb, bit):
                if (dst_index < 8) or (56 <= dst_index):
                    for piece_type in (Piece):
                        if piece_type != Piece.PAWN:
                            moves.append(Move(src_index, dst_index, piece_type, takes))
                else:
                    moves.append(Move(src_index, dst_index, is_take=takes))
            dst_index += bit
            dst_bb = rs(dst_bb, bit)
        return moves
    
    def _get_moves(self):
        # get moves by the self.player_to_move kings bb and returns them as [Move]
        moves:List[Move] = []
        bit = u8(1)
        for piece_type in Piece:
            src_index = uint(0)
            src_bb = self.bitboards[self.player_to_move, piece_type]
            while src_bb:
                if b_and(src_bb, bit):
                    moves += self._get_moves_by_piece_type(piece_type, src_index)
                src_index += bit
                src_bb = rs(src_bb, bit)
        return moves
    
    def _get_moves_by_piece_type(self, piece_type:Piece, src_index:u8):
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
            return self._get_moves_bishop(src_index)
        elif piece_type == Piece.ROOK:
            return self._get_moves_rook(src_index)
        elif piece_type == Piece.QUEEN:
            return self._get_moves_queen(src_index)
        elif piece_type == Piece.KING:
            return self._get_moves_king(src_index)
        else:
            raise RuntimeError("Invalid piece_type: %s" % str(piece_type))

    def _get_moves_file(self, src_index:u8):
        moves:List[Move] = []
        shift = b_and(src_index, u8(0b000_111))
        rank_index = rs(b_and(src_index, u8(0b111_000)), u8(3))

        occ = rs(self.combined[2], shift)
        occ = b_and(occ, LOOKUP.FILE_H_BB)
        occ *= LOOKUP.DIAG_MASKS[7]
        occ = rs(occ, u8(0b111_000))

        dst_bb = LOOKUP.FILE_H_ATTACKS[rank_index, occ]
        dst_bb = ls(dst_bb, shift)

        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_rank(self, src_index:u8):
        moves:List[Move] = []
        rank_index = b_and(src_index, u8(0b000_111))
        shift:u8 = b_and(src_index, u8(0b111_000))

        occ = u8(rs(self.combined[2], shift))

        dst_bb = u64(LOOKUP.FIRST_RANK_ATTACKS[rank_index, occ])
        dst_bb = ls(dst_bb, shift)
        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_diag(self, src_index:u8):
        moves:List[Move] = []
        rank_index = b_and(src_index, u8(0b000_111))

        occ = b_and(self.combined[2], LOOKUP.DIAG_MASKS[src_index])
        occ *= LOOKUP.FILE_H_BB
        occ = rs(occ, u8(0b111_000))

        dst_bb = LOOKUP.FIRST_RANK_ATTACKS[rank_index, occ]
        dst_bb *= LOOKUP.FILE_H_BB
        dst_bb = b_and(dst_bb, LOOKUP.DIAG_MASKS[src_index])

        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_antidiag(self, src_index:u8):
        moves:List[Move] = []
        rank_index = b_and(src_index, u8(0b000_111))

        occ = b_and(self.combined[2], LOOKUP.ANTIDIAG_MASKS[src_index])
        occ *= LOOKUP.FILE_H_BB
        occ = rs(occ, u8(0b111_000))

        dst_bb = LOOKUP.FIRST_RANK_ATTACKS[rank_index, occ]
        dst_bb *= LOOKUP.FILE_H_BB
        dst_bb = b_and(dst_bb, LOOKUP.ANTIDIAG_MASKS[src_index])

        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_pawn_white(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        src_bb = ls(u64(1), src_index)

        takes_bb = b_and(ls(src_bb, u8(9)), b_not(LOOKUP.FILE_H_BB))
        takes_bb = b_or(takes_bb, b_and(ls(src_bb, u8(7)), b_not(LOOKUP.FILE_A_BB)))
        takes_bb = b_and(takes_bb, b_or(self.combined[Color.BLACK], self.enpassante))

        if takes_bb: # if takes avaiable, no need to find non-takes
            return self._get_moves_by_bb_pawn(src_index, takes_bb, takes=True)

        moves_bb = ls(src_bb, u8(8))
        if (8 <= src_index < 16):
            moves_bb = b_or(moves_bb, ls(src_bb, u8(16)))
        moves_bb = b_and(moves_bb, b_not(b_or(self.combined[2], ls(b_and(b_not(src_bb), self.combined[2]), u8(8)))))

        bb = b_or(moves_bb, takes_bb)

        return self._get_moves_by_bb_pawn(src_index, bb)

    def _get_moves_pawn_black(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        src_bb = ls(u64(1), src_index)

        takes_bb = b_and(rs(src_bb, u8(7)), b_not(LOOKUP.FILE_H_BB))
        takes_bb = b_or(takes_bb, b_and(rs(src_bb, u8(9)), b_not(LOOKUP.FILE_A_BB)))
        takes_bb = b_and(takes_bb, b_or(self.combined[Color.WHITE], self.enpassante))

        if takes_bb: # if takes avaiable, no need to find non-takes
            return self._get_moves_by_bb_pawn(src_index, takes_bb, takes=True)

        moves_bb = rs(src_bb, u8(8))
        if (48 <= src_index < 56):
            moves_bb = b_or(moves_bb, rs(src_bb, u8(16)))
        moves_bb = b_and(moves_bb, b_not(b_or(self.combined[2], rs(b_and(b_not(src_bb), self.combined[2]), u8(8)))))

        bb = b_or(moves_bb, takes_bb)

        return self._get_moves_by_bb_pawn(src_index, bb)
    
    def _get_moves_knight(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        moves:List[Move] = []
        dst_bb = LOOKUP.KNIGHT_BB[src_index]
        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))
        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_bishop(self, src_index:u8):
        moves:List[Move] = []
        moves += self._get_moves_diag(src_index)
        moves += self._get_moves_antidiag(src_index)
        return moves

    def _get_moves_rook(self, src_index:u8):
        moves:List[Move] = []
        moves += self._get_moves_rank(src_index)
        moves += self._get_moves_file(src_index)
        return moves
    
    def _get_moves_queen(self, src_index:u8):
        moves:List[Move] = []
        moves += self._get_moves_diag(src_index)
        moves += self._get_moves_antidiag(src_index)
        moves += self._get_moves_rank(src_index)
        moves += self._get_moves_file(src_index)
        return moves

    def _get_moves_king(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        moves:List[Move] = []
        dst_bb = LOOKUP.KING_BB[src_index]
        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))
        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves


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
                        position = b_and(LOOKUP.rank_nr_list[i], LOOKUP.rank_l_list[l])
                        if b_and(piece_type, position) != 0:
                            representation[0][i][l].append(1)
                        else:
                            representation[0][i][l].append(0)

        # TODO represent repetitions in some shape or form for the Chessboard class
        repetitions_w = 0
        repetitions_b = 0

        if self.player_to_move == Color.WHITE:
            color = 0
        else:
            color = 1

        no_progress = self.no_progress_counter[-1]
        en_passant = self.enpassante
        for i in range(0, 8):
            for l in range(0, 8):
                representation[0][i][l].append(repetitions_w)
                representation[0][i][l].append(repetitions_b)
                representation[0][i][l].append(color)
                representation[0][i][l].append(no_progress)
                position = b_and(LOOKUP.rank_nr_list[i], LOOKUP.rank_l_list[l])
                if b_and(en_passant, position) != 0:
                    representation[0][i][l].append(1)
                else:
                    representation[0][i][l].append(0)

        #for row in range(0,8):
            #print(representation[0][row])
        return array(representation)

def print_bb(bb:u64):
    mask_bb = u64(pow(2, 63))
    for i in range(64):
        if not(i%8):
            print()
        if b_and(bb, mask_bb):
            print("1 ", end="")
        else:
            print(". ", end="")
        mask_bb = rs(mask_bb, u8(1))
    print()

def print_byte(byte:u8):
    i = u8(0b10000000)
    while i:
        if byte & i:
            print("1", end="")
        else:
            print("0", end="")
        i >>= u8(1)
    print()