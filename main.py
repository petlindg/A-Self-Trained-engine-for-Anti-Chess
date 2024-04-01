import time

from training import Training
from chess import Chessboard
from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
from training_mp import NeuralNetworkProcess, GameProcess
from multiprocessing import Queue

def run_training():

    chessboard = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")

    nr_workers = 30
    input_queue = Queue()
    output_queues = {}

    worker_list = []
    for i in range(nr_workers):
        output_queues[i] = Queue()
        worker = GameProcess(initial_state=chessboard,
                             input_queue=input_queue,
                             output_queue=output_queues[i],
                             uid=i)
        worker_list.append(worker)

    nn_process = NeuralNetworkProcess(input_queue=input_queue,
                                      output_queues=output_queues
                                      )

    nn_process.daemon = True
    nn_process.start()

    for worker in worker_list:
        worker.daemon = True
        worker.start()

    while True:
        time.sleep(20)

#    training = Training(chessboard, model)
#    training.load_from_file('Game/trainingdata.bz2')
#    print(len(training.buffer))
#    training.train()




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
    run_training()
    #train_file()

if __name__ == '__main__':
    main()
