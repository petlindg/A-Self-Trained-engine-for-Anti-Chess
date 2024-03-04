from training import Training
from chess import Chessboard
from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE

# TODO change the neural network output to sigmoid

def run_training():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except:
        pass
    chessboard = Chessboard("8/8/8/8/7k/8/8/7K w - 0 1")
    training = Training(chessboard, model)
    training.train()


def main():
    run_training()


if __name__ == '__main__':
    main()
