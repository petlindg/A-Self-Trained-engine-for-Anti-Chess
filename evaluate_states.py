import random
from Game.state_generator import generate_random_state
from chess.utils import Piece, Color
from chess.chessboard import Chessboard
from chess.move import algebraic_to_move, Move, calc_move
import pandas as pd
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE
from config import checkpoint_path
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import pickle
from itertools import chain

def fetch_p_from_move_2(move: Move, model_output: np.array):
    """Fetches the P value from the output array of the model

    :param move: Move, move to fetch the P value for
    :param model_output: np.array, array of the model's policy output
    :return: Float, the P value for the move
    """
    src_col = int(move.src_index % 8)
    src_row = int(move.src_index // 8)

    move_type = calc_move(move.src_index, move.dst_index, move.promotion_type)
    #print(model_output)
    model_output = np.array(model_output).reshape(8,8, 79)

    return model_output[src_row][src_col][move_type]

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


def run_testing(model_path):
    # main function to calculate the average loss of the model
    # on the testing dataset in test_states.csv
    df = pd.read_csv('test_states.csv')
    print(df.head())

    # convert the fen string series to the input representation
    df['input_repr'] = df['fen'].apply(calculate_input)

    # create the model
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    try:
        model = tensorflow.keras.models.load_model(model_path)
    except:
        model = model_config.build_nn()
    
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
    
    df['p_value'] = df.apply(lambda x: fetch_p_from_move_2(algebraic_to_move(x.move), x.p), axis=1)
    print(df['p_value'].head())
    df['p_value'].to_csv('p_values.csv')
    df['p_loss'] = df['p_value'].apply(lambda x: 1-x)
    
    print(df['v_loss'].mean())
    print(df['p_loss'].mean())
    return df['v_loss'].mean(), df['p_loss'].mean(), list(df['v_loss']), list(df['p_loss'])

def graph_loss(starting_iteration, ending_iteration):
    p_loss = []
    v_loss = []
    list_losses_v = []
    list_losses_p = []
    for i in range(starting_iteration, ending_iteration+20, 20):
        path = f'saved_model/model_{i}_it.h5'
        v, p, list_v_loss, list_p_loss = (run_testing(path))
        p_loss.append(p)
        v_loss.append(v)
        list_losses_v.append(list_v_loss)
        list_losses_p.append(list_p_loss)
    
    with open('v_losses.pkl', 'wb') as file:
        pickle.dump(list_losses_v, file)

    with open('p_losses.pkl', 'wb') as file:
        pickle.dump(list_losses_p, file)

    plt.plot(range(starting_iteration, ending_iteration+20, 20), p_loss, label='p loss')
    plt.plot(range(starting_iteration, ending_iteration+20, 20), v_loss, label='v loss')
    plt.xlabel('training iterations')
    plt.ylabel('value loss on test set')
    plt.legend()
    plt.title('Value Loss as a function of training games')
    plt.savefig('loss.pdf')
    plt.show()
    plt.clf()

    bins = np.linspace(0, 1, 100)
    plt.hist(list_losses_v[0], bins, alpha=0.5, label='iteration 20')
    plt.hist(list_losses_v[6], bins, alpha=0.5, label='iteration 140')
    plt.legend()
    plt.savefig('hist_loss.pdf')

def plot_hist(start_it, step_size):
    with open('v_losses.pkl', 'rb') as file: 
        list_v_losses = pickle.load(file) 

    list_heights = []
    
    for i, l in enumerate(list_v_losses):
        list_heights.append([start_it+i*step_size]*len(l))
    height_length = len(list_heights)
    list_v_losses = list(chain.from_iterable(list_v_losses))
    list_heights = list(chain.from_iterable(list_heights))


    #for x, y in zip(list_losses, list_heights):
    fig, ax = plt.subplots()
    h = ax.hist2d(list_v_losses, list_heights, bins=[20,height_length], norm='log')
    #plt.hist2d(list_losses, list_heights, bins=[50,height_length], norm='log')
    fig.colorbar(h[3], ax=ax)

    plt.ylabel('Iteration (each iteration is 50 games played)')
    plt.xlabel('Distribution of squared loss values (log color scale)')
    plt.title('Loss Distribution per iteration for v')
    fig.savefig('hist2d_loss_v.pdf')

    with open('p_losses.pkl', 'rb') as file: 
        list_p_losses = pickle.load(file) 
        list_heights = []
    
    for i, l in enumerate(list_p_losses):
        list_heights.append([start_it+i*step_size]*len(l))
    height_length = len(list_heights)
    list_p_losses = list(chain.from_iterable(list_p_losses))
    list_heights = list(chain.from_iterable(list_heights))

    

    #for x, y in zip(list_losses, list_heights):
    fig, ax = plt.subplots()
    h = ax.hist2d(list_p_losses, list_heights, bins=[20,height_length], norm='log')
    #plt.hist2d(list_losses, list_heights, bins=[50,height_length], norm='log')
    fig.colorbar(h[3], ax=ax)

    plt.ylabel('Iteration (each iteration is 50 games played)')
    plt.xlabel('Distribution of loss values (log color scale)')
    plt.title('Loss Distribution per iteration for p')
    fig.savefig('hist2d_loss_p.pdf')

def main():
    #get_randomized_states(10000)
    graph_loss(0, 300)
    #plot_hist(0, 20)
if __name__ == '__main__':
    main()