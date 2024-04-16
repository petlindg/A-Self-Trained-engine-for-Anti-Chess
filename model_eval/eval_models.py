import sys
sys.path.append('..')

from multiprocessing import Queue, set_start_method
import time
from keras import Model
import tensorflow

from chess.chessboard import Chessboard
from model_eval.game_process_eval import GameProcessEval
from model_eval.nn_process_eval import NeuralNetworkProcessEval

EVAL_GAMES = 100

def eval_models(model_1:Model, model_2:Model, games:int=100, initial_state:Chessboard=Chessboard()):
    nn_batch = games//2
    set_start_method('spawn')

    nr_workers = games

    input_queue_1 = Queue()
    output_queues_1 = {}
    input_queue_2 = Queue()
    output_queues_2 = {}

    worker_list = []
    for i in range(nr_workers):
        # for every worker, create their personal queue and the process
        output_queues_1[i] = Queue()
        output_queues_2[i] = Queue()
        worker = GameProcessEval(initial_state=initial_state,
                                input_queue_1=input_queue_1,
                                output_queue_1=output_queues_1[i],
                                input_queue_2=input_queue_2,
                                output_queue_2=output_queues_2[i],
                                uid=i)
        worker_list.append(worker)



    nn_process_1 = NeuralNetworkProcessEval(input_queue=input_queue_1,
                                            output_queues=output_queues_1,
                                            nn_batch=nn_batch,
                                            model = model_1
                                            )
    nn_process_1.daemon = True
    nn_process_1.start()

    nn_process_2 = NeuralNetworkProcessEval(input_queue=input_queue_2,
                                            output_queues=output_queues_2,
                                            nn_batch=nn_batch,
                                            model = model_2
                                            )
    nn_process_2.daemon = True
    nn_process_2.start()

    for worker in worker_list:
        worker.daemon = True
        worker.start()

    while True:
        time.sleep(20)





def main():
    model_1 = tensorflow.keras.models.load_model('../saved_model/model_40_it.h5', compile=False)
    model_1.compile()
    model_2 = tensorflow.keras.models.load_model('../saved_model/model_60_it.h5', compile=False)
    model_1.compile()
    eval_models(model_1, model_2, 2)

if __name__=="__main__":
    main()