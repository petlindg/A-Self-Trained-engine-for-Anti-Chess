import os
import time

import tensorflow

import config
from training import Training
from chess.chessboard import Chessboard
from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
from nn_process import NeuralNetworkProcess
from game_process import GameProcess
from multiprocessing import Queue, set_start_method


def run_training(fen, workers=1):
    """
    Function that runs the training for a given fen notation,
    workers is an argument for how many game processes that will be run at the same time
    """
    set_start_method('spawn')

    nr_workers = workers
    input_queue = Queue()
    output_queues = {}

    worker_list = []
    for i in range(nr_workers):
        # for every worker, create their personal queue and the process
        output_queues[i] = Queue()
        worker = GameProcess(initial_state=fen,
                             input_queue=input_queue,
                             output_queue=output_queues[i],
                             uid=i)
        worker_list.append(worker)

    # create the nn_process and give it all of the queues
    nn_process = NeuralNetworkProcess(input_queue=input_queue,
                                      output_queues=output_queues
                                      )

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

    # number of workers/threads to train with
    threads = config.processes
    # if evaluation is active, we only run a single thread
    if config.evaluation:
        threads = 1

    run_training("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1", threads)
    #train_file()

if __name__ == '__main__':
    main()
