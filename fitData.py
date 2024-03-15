import pickle

from training import Training
from chess import Chessboard
from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE

def main():
    fp = "trainingData/training_data_2024-03-13T15"
    file = open(fp, 'rb')
    game_history = pickle.load(file)
    
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    chessboard = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
    
    trainer = Training(chessboard, model)
    trainer.buffer.append(game_history)
    trainer.fit_data()
    model.save_weights(checkpoint_path)

if __name__=="__main__":
    main()