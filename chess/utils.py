from numpy import uint64 as u64
from numpy import uint8 as u8
from numpy import right_shift as rs
from numpy import bitwise_and as b_and

from enum import IntEnum

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

def alg_sq_to_index(alg:str):
        file_char = alg[0]
        rank_char = alg[1]
        return 72-ord(file_char) + (int(rank_char)-1)*8

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