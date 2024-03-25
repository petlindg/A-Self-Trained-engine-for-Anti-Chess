"""
Module for testing performance of move generation.
"""

import timeit as t

from chessboard import Chessboard

DEPTH = 3

def perft(cb:Chessboard, depth:int):
    global counter
    if depth==0:
        return
    moves = cb.get_moves()
    for m in moves:
        cb.move(m)
        perft(cb, depth-1)
        cb.unmove()


def main():

    cb = Chessboard()
    print(t.timeit(stmt=lambda: perft(cb, DEPTH), number=1))

if __name__ == "__main__":
    main()