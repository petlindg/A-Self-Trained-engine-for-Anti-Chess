from numpy import uint64 as u64
from numpy import uint8 as u8
from numpy import left_shift as ls
from numpy import right_shift as rs
from numpy import bitwise_or as b_or
from numpy import bitwise_and as b_and
from numpy import bitwise_not as b_not
from numpy import bitwise_xor as b_xor
from numpy import zeros, ndarray, uint, array

RANK_1_BB = u64(0b11111111)
RANK_2_BB = u64(0b11111111_00000000)
RANK_3_BB = u64(0b11111111_00000000_00000000)
RANK_4_BB = u64(0b11111111_00000000_00000000_00000000)
RANK_5_BB = u64(0b11111111_00000000_00000000_00000000_00000000)
RANK_6_BB = u64(0b11111111_00000000_00000000_00000000_00000000_00000000)
RANK_7_BB = u64(0b11111111_00000000_00000000_00000000_00000000_00000000_00000000)
RANK_8_BB = u64(0b11111111_00000000_00000000_00000000_00000000_00000000_00000000_00000000)

rank_nr_list: list[u64] = [RANK_8_BB, RANK_7_BB, RANK_6_BB, RANK_5_BB, RANK_4_BB, RANK_3_BB, RANK_2_BB, RANK_1_BB]

FILE_A_BB = u64(0b10000000_10000000_10000000_10000000_10000000_10000000_10000000_10000000)
FILE_B_BB = u64(0b01000000_01000000_01000000_01000000_01000000_01000000_01000000_01000000)
FILE_C_BB = u64(0b00100000_00100000_00100000_00100000_00100000_00100000_00100000_00100000)
FILE_D_BB = u64(0b00010000_00010000_00010000_00010000_00010000_00010000_00010000_00010000)
FILE_E_BB = u64(0b00001000_00001000_00001000_00001000_00001000_00001000_00001000_00001000)
FILE_F_BB = u64(0b00000100_00000100_00000100_00000100_00000100_00000100_00000100_00000100)
FILE_G_BB = u64(0b00000010_00000010_00000010_00000010_00000010_00000010_00000010_00000010)
FILE_H_BB = u64(0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001)

rank_l_list: list[u64] = [FILE_A_BB, FILE_B_BB, FILE_C_BB, FILE_D_BB, FILE_E_BB, FILE_F_BB, FILE_G_BB, FILE_H_BB]

def _rank_masks_init():
    arr = zeros(64, dtype=u64)
    for index in range(64):
        if not index%8:
            rank = ls(RANK_1_BB, u8(index))
        arr[index] = rank
    return arr
def _file_masks_init():
    arr = zeros(64, dtype=u64)
    for index in range(64):
        if not index%8:
            file = FILE_H_BB
        else:
            file = ls(file, u8(1))
        arr[index] = file
    return arr
def _diag_masks_init():
    arr = zeros(64, dtype=u64)
    bit = u64(1)
    for index in range(64):
        bb = u64(0)
        sq = ls(bit, u64(index))
        while(sq):
            bb = b_or(bb, sq)
            sq = b_and(ls(sq, uint(7)), b_not(FILE_A_BB))
        sq = ls(bit, u64(index))
        while(sq):
            bb = b_or(bb, sq)
            sq = b_and(rs(sq, uint(7)), b_not(FILE_H_BB))
        arr[index] = bb
    return arr
def _antidiag_masks_init():
    arr = zeros(64, dtype=u64)
    bit = u64(1)
    for index in range(64):
        bb = u64(0)
        sq = ls(bit, u64(index))
        while(sq):
            bb = b_or(bb, sq)
            sq = b_and(ls(sq, uint(9)), b_not(FILE_H_BB))
        sq = ls(bit, u64(index))
        while(sq):
            bb = b_or(bb, sq)
            sq = b_and(rs(sq, uint(9)), b_not(FILE_A_BB))
        arr[index] = bb
    return arr
def _calc_first_rank_attacks(index:u8, occ:u8):
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
def _calc_file_h_attacks(index:u8, occ:u8):
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
def _first_rank_attacks_init():
    arr = zeros((8, 256), dtype=u8)
    for index in range(8):
        for occ in range(256):
            arr[index, occ] = _calc_first_rank_attacks(index, occ)
    return arr
def _file_h_attacks_init():
    arr = zeros((8, 256), dtype=u64)
    for index in range(8):
        for occ in range(256):
            arr[index, occ] = _calc_file_h_attacks(index, occ)
    return arr
def _knight_bb_init():
    # Init of movegeneration bitboards for knights.
    # The bitboard of bbs[index] represents the bitboard with all possible destinationsquares given a source square = index
    bbs = zeros(64, dtype=u64)
    src_bb = u64(1) # source bb to generate moves from
    for index in range(u64(64)):
        dst_bb = u64(0) # destination bb to track all possible move destinations
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(17)), b_not(FILE_A_BB)))
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(15)), b_not(FILE_H_BB)))
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(10)), b_not(b_or(FILE_A_BB, FILE_B_BB))))
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(6)),  b_not(b_or(FILE_G_BB, FILE_H_BB))))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(6)),  b_not(b_or(FILE_A_BB, FILE_B_BB))))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(10)), b_not(b_or(FILE_G_BB, FILE_H_BB))))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(15)), b_not(FILE_A_BB)))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(17)), b_not(FILE_H_BB)))
        bbs[index] = dst_bb
        src_bb = ls(src_bb, u8(1)) # shift src_bb to match index
    return bbs
def _king_bb_init():
    # Init of movegeneration bitboards for kings.
    # The bitboard of bbs[index] represents the bitboard with all possible destinationsquares given a source square = index
    bbs = zeros(64, dtype=u64)
    src_bb = u64(1) # source bb to generate moves from
    for index in range(u64(64)):
        dst_bb = u64(0) # destination bb to track all possible move destinations
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(9)), b_not(FILE_A_BB)))
        dst_bb = b_or(dst_bb,       rs(src_bb, u8(8)))
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(7)), b_not(FILE_H_BB)))
        dst_bb = b_or(dst_bb, b_and(rs(src_bb, u8(1)), b_not(FILE_A_BB)))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(1)),  b_not(FILE_H_BB)))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(7)),  b_not(FILE_A_BB)))
        dst_bb = b_or(dst_bb,       ls(src_bb, u8(8)))
        dst_bb = b_or(dst_bb, b_and(ls(src_bb, u8(9)),  b_not(FILE_H_BB)))
        bbs[index] = dst_bb
        src_bb = ls(src_bb, u8(1)) # shift src_bb to match index
    return bbs

KNIGHT_BB = zeros(64, dtype=u64)
KING_BB = zeros(64, dtype=u64)
FIRST_RANK_ATTACKS = _rank_masks_init()
FILE_H_ATTACKS = _file_masks_init()
DIAG_MASKS = _diag_masks_init()
ANTIDIAG_MASKS = _antidiag_masks_init()
FIRST_RANK_ATTACKS = _first_rank_attacks_init()
FILE_H_ATTACKS = _file_h_attacks_init()
KNIGHT_BB = _knight_bb_init()
KING_BB = _king_bb_init()