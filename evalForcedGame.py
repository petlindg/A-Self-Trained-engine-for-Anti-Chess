
from chess.chessboard import Chessboard
from chess.move import algebraic_to_move
from config import *
from Model.nn_architecture import INPUT_SHAPE, OUTPUT_SHAPE, NeuralNetwork
from MCTS.node import fetch_p_from_move

def possible_moves(state:Chessboard, model):
    # copied from node.py, should be moved somewhere else to be called outside of a MCTS tree
    """Calculates all possible moves for a given chessboard using the neural network, and returns
       it as a list of tuples.
    :param state: Chessboard, the input state to calculate moves from
    :return: (list[Chessboard, Move, float], float), returns a list of new chessboards, the move
             that was taken to get there and the float value P for that new state. In addition to this,
             it also returns another float which is the value V from the neural network for the input state.
    """
    input_repr = state.translate_board()
    moves = state.get_moves()
    p, v = model.predict(input_repr, verbose=None)
    v = v[0][0]
    p_array = p.reshape(output_representation)
    return_list = []
    p_sum = 0
    for move in moves:
        p_val = fetch_p_from_move(move, p_array)
        p_sum += p_val
        return_list.append((move, p_val))
    # normalize the P values in the return list
    return_list = [(move, p_val/p_sum) for (move, p_val) in return_list]

    return return_list, v, 1-p_sum

def eval_forced_Game(fen:str, moves_alg:str):
    """
    Plays a game given a fen and a list of moves seperated by spaces and prints the models evaluation
    of every state in the given game
    :param fen: FEN string representation of initial state
    :param moves_alg: algebraic move representation for the game, seperated by spaces
    """

    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        print("Loading existing weights...")
        model.load_weights(checkpoint_path)
    except:
        raise RuntimeError("No existing weights for model.") #weights should always exist when trying to evaluate game
    moves = [algebraic_to_move(move_alg) for move_alg in moves_alg.split()]
    chessboard = Chessboard(fen)
    for move in moves:
        print(chessboard)
        p_list, v, p_ill= possible_moves(chessboard, model)
        print(f"v: {v}")
        print(f"p_ill:{p_ill}")
        for (m, p) in p_list:
            print(f"move: {m}, p:{p}")
        chessboard.move(move)



def main():
    eval_forced_Game("k7/8/8/8/8/8/8/7R w - 0 1", "h1c1 a8a7 c1c5 a7a8 c5c6 a8b8 c6c7 b8c7")

if __name__=="__main__":
    main()
