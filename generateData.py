import bz2
import copy
from Game.Utils import translate_moves_to_output
from chess import Move
from chess import algebraic_to_move
from chess import Chessboard
from chess import Color
from config import *
from copy import deepcopy
from node import Node
from datetime import datetime
import time

import pickle
import sys

def generate_training_data(fen:Chessboard,
                           movesAlg:str,
                           fileName:str="",
                           fp:str=TRAINING_DATA_PATH,
                           iterations:int=tree_iterations):
    """
    Function to generate training data and save it on drive in , given a sequence of legal moves.

    :param fen: The FEN board representation of initial state to generate data from
    :param movesAlg: Sequence of moves given in algebraic chess notation seperated by space.
    :param fileName: Name to save training data as, formatted as training_data_{fileName}
    :param iterations: Number of tree iterations to run. Defaults to tree_iterations from config.py
    :param fp: File path to save training data in. Defaults to TRAINING_DATA_PATH from config.py
    :return: None
    """

    data = []
    try:
        with bz2.BZ2File('trainingdata.bz2', 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        print(e)

    state = Chessboard(fen)
    tree = Node(state=state)
    moves = [algebraic_to_move(m) for m in movesAlg.split()]
    game_history = []
    training_data = []

    while moves:
        tree.run(iterations)
        p_list = [c.visits/tree.visits for c in tree.children]
        game_history.append((deepcopy(state), (zip(p_list, tree.state.get_moves()))))
        tree = tree.update_tree(moves.pop(0))

    status = tree.state.get_game_status()
    if status == 0:
        winner = Color.WHITE
    elif status == 1:
        winner = Color.BLACK
    elif status == 2:
        winner = 0.5
    else:
        raise RuntimeError("Game still on going.")
    
    for (state, p_list) in game_history:
        if state.player_to_move == winner:
            value = 1
        elif state.not_player_to_move == winner:
            value = 0
        else:
            value = 0.5
        training_data.append((state.translate_board(), translate_moves_to_output(p_list), value)) # 0 here instead of Node, should reconsider get_history from Game
        print(state)
        for (p, move) in p_list:
            print(p, move)
        
        data = [training_data]

    with bz2.BZ2File('trainingdata.bz2', 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) >= 5:
        # grab given values
        fen = str(sys.argv[1])
        movesAlg = str(sys.argv[2])
        fileName = str(sys.argv[3])
        iterations = int(sys.argv[4])
        generate_training_data(fen, movesAlg, fileName, iterations=iterations)
    else:
        # else use default values
        fen = "k7/8/8/8/8/8/8/7R w - 0 1"
        movesAlg = "h1c1 a8a7 c1c5 a7a8 c5c6 a8b8 c6c7 b8c7"
        generate_training_data(fen, movesAlg)

if __name__=='__main__':
    main()