import bz2
import pickle

from chess.utils import Color, Piece
from config import *

# TODO
# Print P in readable format

def print_bb_arr(rep, index):
    """
    Prints a plane of the nn input representation given an index
    :param rep: chessboard as input representation for our network (List of shape (1, 8, 8, 17))
    :param index: index of which plane to print
    """
    for i in range(8):
        for j in range(8):
            print(f"{rep[0][i][j][index]} ", end="")
        print("")
            

def print_rep(rep):
    """
    Prints the input representation of the NN in a readable format
    :param rep: chessboard as input representation for our network (List of shape (1, 8, 8, 17))
    """
    for c in Color:
        for p in Piece:
            print(f"Color:{c}, Piece:{p}")
            print_bb_arr(rep, c*6+p)

    print("repetitions_1:")
    print_bb_arr(rep, 12)
    
    print("repetitions_2:")
    print_bb_arr(rep, 13)

    print("color:")
    print_bb_arr(rep, 14)

    print("no_progress:")
    print_bb_arr(rep, 15)

    print("enpassant:")
    print_bb_arr(rep, 16)

def print_data_set(dataset):
    """
    Prints a dataset in a readable format
    :param dataset: Dataset to print
    """
    for game in dataset:
        for (state, p_list, winner) in game:
            print_rep(state)
            print(winner)

def main():
    data = []
    try:
        with bz2.BZ2File(f"{TRAINING_DATA_PATH}/trainingdata.bz2", 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        print(e)
    print_data_set(data)

if __name__=='__main__':
    main()