from training import Training
from chess import Chessboard
from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE

from Game.Game import Game
from Game.CliPlayer import CliPlayer
from Game.EnginePlayer import EnginePlayer
from node import Node
from copy import deepcopy

# TODO change the neural network output to sigmoid

def run_training():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)
    chessboard = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
    training = Training(chessboard, model)
    training.load_from_file('Game/trainingdata.bz2')
    print(len(training.buffer))
    training.train()

def train_file():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        pass
        #model.load_weights(checkpoint_path)
    except:
        pass
    chessboard = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
    training = Training(chessboard, model)
    training.train_from_file('Game/trainingdata.bz2')

def main():
    #run_training()
    # train_file()

    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except:
        pass
    chessboard = Chessboard("8/3r4/2kkk3/8/8/2KKK3/3R4/8 w - 0 1")

    game = Game(chessboard, EnginePlayer(deepcopy(chessboard), model), CliPlayer(deepcopy(chessboard)))
    result = game.run()

    print(result)

if __name__ == '__main__':
    main()
