from itertools import count
import sys
sys.path.append('..')

from multiprocessing import Queue, set_start_method
import time
from keras import Model
import tensorflow
from node import Node
from chess.chessboard import Chessboard
from model_eval.game_process_eval import GameProcessEval
from model_eval.nn_process_eval import NeuralNetworkProcessEval

EVAL_GAMES = 100

def eval_models(model_1:Model, model_2:Model, games:int=100, initial_state:Chessboard=Chessboard()):
    """
    Evaluates the win-rate, draw-rate and move counters of a set number of games between two models.

    :param model_1: The first model to be evaluated
    :param model_2: The second model to be evaluated
    :param games: Number of games to be played between the models
    :param initial_state: The state of which the games should be played
    """
    nn_batch = games//2

    set_start_method('spawn')

    results_queue = Queue()

    nr_workers = games

    # initiate the queues to be used between the GameProcessEval instances and the NeuralNetworkProcessEval instances
    input_queue_1 = Queue()
    output_queues_1 = {}
    input_queue_2 = Queue()
    output_queues_2 = {}


    # initiate the GameProcessEvals
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
                                results_queue=results_queue,
                                uid=i)
        worker_list.append(worker)

    # initiate the first NeuralNetworkProcessEval and start it
    nn_process_1 = NeuralNetworkProcessEval(input_queue=input_queue_1,
                                            output_queues=output_queues_1,
                                            nn_batch=nn_batch,
                                            model = model_1
                                            )
    nn_process_1.daemon = True
    nn_process_1.start()

    # initiate the second NeuralNetworkProcessEval and start it
    nn_process_2 = NeuralNetworkProcessEval(input_queue=input_queue_2,
                                            output_queues=output_queues_2,
                                            nn_batch=nn_batch,
                                            model = model_2
                                            )
    nn_process_2.daemon = True
    nn_process_2.start()

    # start the GameProcessEvals
    for worker in worker_list:
        worker.daemon = True
        worker.start()

    # add results to list
    results = []
    while len(results)<games:
        r = results_queue.get()
        results.append(r)
        
    # process data
    win_count_model_1 = [winner for (uid, winner, move_count, model_1_p, model_2_p) in results].count("model_1")
    win_rate_model_1 = win_count_model_1/games
    win_count_model_2 = [winner for (uid, winner, move_count, model_1_p, model_2_p) in results].count("model_2")
    win_rate_model_2 = win_count_model_2/games
    draw_count = [winner for (uid, winner, move_count, model_1_p, model_2_p) in results].count("draw")
    draw_rate = draw_count/games
    avg_p_legal_model_1 = sum([model_1_p for (uid, winner, move_count, model_1_p, model_2_p) in results])/games
    avg_p_legal_model_2 = sum([model_2_p for (uid, winner, move_count, model_1_p, model_2_p) in results])/games
    move_avg = sum([move_count for (uid, winner, move_count, model_1_p, model_2_p) in results])/games

    return (win_count_model_1, win_count_model_2, draw_count, avg_p_legal_model_1, avg_p_legal_model_2, move_avg)

def main():
    # needs to run from model_eval folder
    model_1 = tensorflow.keras.models.load_model('../saved_model/model_20_it.h5', compile=False)
    model_1.compile()
    model_2 = tensorflow.keras.models.load_model('../saved_model/model_420_it.h5', compile=False)
    model_2.compile()

    win_count_model_1, win_count_model_2, draw_count, avg_p_legal_model_1, avg_p_legal_model_2, move_avg = eval_models(model_1, model_2, 10)
    print(f"Result distribution [model 1|model 2|draw]: [{win_count_model_1}|{win_count_model_2}|{draw_count}]")
    print(f"Average legal P values generated [model 1|model 2]: [{avg_p_legal_model_1}|{avg_p_legal_model_2}]")
    print(f"Average amount of moves: [{move_avg}]")

if __name__=="__main__":
    main()