from numpy import uint64 as u64
from numpy import uint8 as u8
from numpy import left_shift as ls
from numpy import right_shift as rs
from numpy import bitwise_or as b_or
from numpy import bitwise_and as b_and
from numpy import bitwise_not as b_not
from numpy import bitwise_xor as b_xor
from numpy import zeros, ndarray, uint, array

from itertools import chain

from utils import Color
from utils import Piece

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

    src_index = u8(-src_col + 8 * fst_d)
    dst_index = u8(-dst_col + 8 * snd_d)
    return Move(src_index, dst_index)

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
    def __eq__(self, move):
        return self.src_index == move.src_index and self.dst_index == move.dst_index and self.promotion_type == move.promotion_type