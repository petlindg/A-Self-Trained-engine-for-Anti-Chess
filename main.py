
from training import Training
import chess
from chess import Chessboard
from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE


def run_training():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except:
        pass
    chessboard = Chessboard()
    chessboard.init_board_test_2()
    training = Training(chessboard, model)
    training.train()

def translate_moves_to_output(mcts_dist):
    """Translates a list of moves into the output representation for the neural network

    :param mcts_dist: list of tuples [(value, move)], value is the visit % for the corresponding move.
    :return: array of shape (1x1x8x8x76), the shape of the output from the neural network.
    """
    output = [[
        [[[0 for i in range(76)] for i in range(8)] for i in range(8)]
    ]]
    # fetch all the available moves
    # for every move, calculate what type value it has and set
    # the array index as 1 for the given move
    for (val, move) in mcts_dist:
        src_col = move.src_index % 8
        src_row = move.src_index // 8
        type_value = chess.calc_move(move.src_index, move.dst_index, move.promotion_type)
        output[0][0][src_row][src_col][type_value] = val

    # return all the moves in output representation
    return output

def main():
    run_training()


if __name__ == '__main__':
    main()