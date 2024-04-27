import random
from Game.state_generator import generate_random_state
from chess.utils import Piece, Color
from chess.chessboard import Chessboard
import pandas as pd
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
from config import checkpoint_path
import numpy as np
from node import fetch_p_from_move


def randomize_state():
    # first randomize the piece for white and the piece for black
    # then randomize the states and check for results from chess.py
    # if there is 1 possible move, perform it, if the game has ended then save the game
    # if not then discard the game and try again
    w_piece = (Color.WHITE, Piece(random.randint(0, 5)))
    b_piece = (Color.BLACK, Piece(random.randint(0, 5)))
    
    random_fen = generate_random_state([w_piece, b_piece])
    game = Chessboard(random_fen)
    # if there is only one move available it means that its a 'forced' move or a pawn move
    # if there are more than 1 moves available it isnt an end state, therefore try again
    if len(game.get_moves()) == 1:
        m = game.get_moves()[0]
        game.move(game.get_moves()[0])
        if game.get_game_status() != 3:
            return [random_fen, game.get_game_status(), m]
    else:
        return None

def get_randomized_states(states_requested):
    # get several randomized states, creates as many states as the argument
    games = []
    while len(games) < states_requested:
        game = randomize_state()
        if game is not None:
            games.append(game)

    # save the game to a csv
    df = pd.DataFrame(games, columns=['fen','status', 'move'])
    df.to_csv('test_states.csv', index=False)

def calculate_input(fen_not):
    # small function for use with pd.apply()
    return np.asarray(Chessboard(fen_not).translate_board()).astype('float32')[0]


def run_testing():
    # main function to calculate the average loss of the model
    # on the testing dataset in test_states.csv
    df = pd.read_csv('test_states.csv')
    print(df.head())

    # convert the fen string series to the input representation
    df['input_repr'] = df['fen'].apply(calculate_input)

    # create the model
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)
    
    # convert the pandas series to numpy array
    states = np.asarray(list(df['input_repr']))
    
    # predict on the states.
    prediction_result = model.predict(states, batch_size=64)

    # extract the values and format them correctly
    v_list = prediction_result[1].tolist()
    v_list = [value[0] for value in v_list]

    # extract the p values TODO: implement p loss
    p_list = prediction_result[0].tolist()
    # add the values and p to the pandas dataframe
    df['value'] = pd.Series(v_list)
    df['p'] = pd.Series(p_list)
    
    # calculating the value loss squared
    df['v_loss'] = (df['status'] - df['value']).apply(lambda x: x*x)
    
    #df['p_value'] = df.apply(lambda x: fetch_p_from_move(x.move, x.p), axis=1)
    print(df.head())
    print(df['v_loss'].mean())
    
def main():
    #get_randomized_states(1000)
    run_testing()

if __name__ == '__main__':
    main()