

import os

import tensorflow

from Game.CliPlayer import CliPlayer
from Game.EnginePlayer import EnginePlayer
from Game.Game import Game
from chess import Chessboard
from config import checkpoint_path
from nn_architecture import INPUT_SHAPE, OUTPUT_SHAPE, NeuralNetwork
from node import Node


def playModel():
    model = tensorflow.keras.models.load_model('saved_model/model_300_it.h5', compile=False)
    model.compile()
    chessboard = Chessboard("8/8/8/3k4/8/8/8/7R w - 0 1")
    player = CliPlayer(chessboard=chessboard)
    engine = EnginePlayer(chessboard, model)
    game = Game(chessboard, player, engine)
    game.run()


def main():
    playModel()

if __name__=="__main__":
    main()