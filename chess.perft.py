"""
Module for testing performance of move generation.
"""

DEPTH = 3

from chess import *
import timeit as t

def perft(cb:Chessboard, depth:int):
    if depth==0:
        return
    moves = cb.get_moves()
    for m in moves:
        cb.move(m)
        perft(cb, depth-1)
        cb.unmove()


def main():

    cb = Chessboard()
    cb.init_board_standard()
    print(t.timeit(stmt=lambda: perft(cb, DEPTH), number=1))

if __name__ == "__main__":
    main()