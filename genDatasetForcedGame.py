import bz2
from Game.Utils import translate_moves_to_output
from chess.move import algebraic_to_move
from chess.chessboard import Chessboard
from chess.utils import Color
from config import *
from copy import deepcopy
from node import Node

import pickle
from printDataset import print_data_set

def gen_dataset_forced_game(fen:Chessboard,
                           movesAlg:str,
                           fileName:str="",
                           fp:str=TRAINING_DATA_PATH,
                           iterations:int=tree_iterations,
                           verbosity=True):
    """
    Function to generate training data and save it on drive in , given a sequence of legal moves.

    :param fen: The FEN board representation of initial state to generate data from
    :param movesAlg: Sequence of moves given in algebraic chess notation seperated by space.
    :param fileName: Name to save training data as, formatted as training_data_{fileName}
    :param iterations: Number of tree iterations to run. Defaults to tree_iterations from config.py
    :param fp: File path to save training data in. Defaults to TRAINING_DATA_PATH from config.py
    :param verbosity: Prints generating process if set to True
    :return: None
    """

    state = Chessboard(fen)
    tree = Node(state=state)
    moves = [algebraic_to_move(m) for m in movesAlg.split()]
    game_history = []
    training_data = []

    while moves:
        tree.run(iterations)
        p_list = [(c.visits/tree.visits, c.move) for c in tree.children]
        game_history.append((deepcopy(state), p_list))
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
    data = [training_data]

    if verbosity:
        print_data_set(data)

    with bz2.BZ2File(f"{TRAINING_DATA_PATH}/trainingdata.bz2", 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main():
    gen_dataset_forced_game("k7/8/8/8/8/8/8/7R w - 0 1", "h1c1 a8a7 c1c5 a7a8 c5c6 a8b8 c6c7 b8c7")

if __name__=='__main__':
    main()
