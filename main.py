import time

import config
from Game.game import Game
from Model.local_model import LocalModel
from Model.remote_model import RemoteModel
from Player.cli_player import CliPlayer
from Player.engine_player import EnginePlayer
from testing import Testing
from training import Training
from chess.chessboard import Chessboard
from Model.nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
from Multithreading.nn_process import NeuralNetworkProcess
from Multithreading.game_process import GameProcess
from multiprocessing import Queue, set_start_method


def run_training(fen, workers=1):
    """
    Function that runs the training for a given fen notation,
    workers is an argument for how many game processes that will be run at the same time
    """
    set_start_method('spawn')
    nr_workers = workers

    # create the nn_process and give it all of the queues
    nn_process = NeuralNetworkProcess()

    worker_list = []
    for i in range(nr_workers):
        player = EnginePlayer(Chessboard(fen), RemoteModel(nn_process.create_connection(i)))
        worker = GameProcess(initial_state=fen, player_1=player)
        worker_list.append(worker)

    # start the neural network as a daemon
    nn_process.daemon = True
    nn_process.start()

    # start all of the workers as daemons
    for worker in worker_list:
        worker.daemon = True
        worker.start()

    # let this main thread sleep (can be changed in the future to do something productive)
    # once the main thread exits, the workers and nn will exit as well.
    while True:
        time.sleep(20)


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
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = LocalModel(model_config.build_nn())

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1"
    game = Game(Chessboard(fen), EnginePlayer(Chessboard(fen), model), CliPlayer(Chessboard(fen)))
    result = game.run()

    print(result)
    # test = Testing("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1", model, model)
    # test.test(1)

    # number of workers/threads to train with
    # threads = config.processes
    # if evaluation is active, we only run a single thread
    # if config.evaluation:
    #   threads = 1

    #run_training("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1", threads)
    #train_file()

if __name__ == '__main__':
    main()
