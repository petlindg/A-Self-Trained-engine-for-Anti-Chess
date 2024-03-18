
from chess import Chessboard
from chess import Move
from chess import algebraic_to_move
from config import *
from nn_architecture import INPUT_SHAPE, OUTPUT_SHAPE, NeuralNetwork
from node import fetch_p_from_move

def possible_moves(state:Chessboard, model):
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

    return return_list, v

def print_eval(return_list, v):
    print(f"v: {v}")
    for (move, p) in return_list:
        print(f"Move: {move}, p:{p}")

def main():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except:
        raise RuntimeError("No existing weights for model.")
    moves_alg = "h1c1 a8a7 c1c5 a7a8 c5c6 a8b8 c6c7 b8c7"
    moves = [algebraic_to_move(move_alg) for move_alg in moves_alg.split()]
    chessboard = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
    for move in moves:
        print_eval(possible_moves(chessboard, model))
        chessboard.move(move)






if __name__=="__main__":
    main()