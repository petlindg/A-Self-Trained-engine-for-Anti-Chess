

import time

import numpy as np
import tensorflow as tf
from chess import Chessboard
from config import *
from nn_revised import INPUT_SHAPE, OUTPUT_SHAPE, NeuralNetwork

BATCH_SIZE = 100

def testPredictSingle():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)

    cb = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
    

    data = []
    for _ in range(BATCH_SIZE):
        data.append(cb.translate_board())

    time_start = time.time_ns()
    for i in range(BATCH_SIZE):
        model.predict(data[i], verbose=None)
    time_end = time.time_ns()
    return time_end-time_start

def testPredictBatch():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)

    cb = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
    

    data = []
    for _ in range(BATCH_SIZE):
        data.append(cb.translate_board())

    time_start = time.time_ns()
    model.predict(np.concatenate(data, axis=0), verbose=None)
    time_end = time.time_ns()
    return time_end-time_start

def testCall():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)

    cb = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")

    input_rep = cb.translate_board()
    time_start = time.time_ns()
    for _ in range(BATCH_SIZE):
        model(input_rep, training=False)
    time_end = time.time_ns()
    return time_end-time_start

def main():
    print(f"Predict single time: {testPredictSingle()/1000000000} seconds")
    print(f"Predict batch time: {testPredictBatch()/1000000000} seconds")
    print(f"Call time: {testCall()/1000000000} seconds")

if __name__=="__main__":
    main()