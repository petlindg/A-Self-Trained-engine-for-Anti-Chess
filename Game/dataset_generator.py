import time
import sys
sys.path.append('..')
import config
from TrainingGame import TrainingGame
from chess import Chessboard, Color
import pickle
import bz2
from state_generator import generate_random_state


def load_data_file():
    try:
        with bz2.BZ2File('trainingdata.bz2', 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        print(e)
        data = []
    return data

def run_games(original_fen, random_state=False, load_data=False, games_played=100):
    """A function that takes a fen notation string and runs games of that fen state
    without the neural network, has an additional parameter for whether to randomize the states
    where the randomized board is based on the config.piece_list

    :param original_fen: String, Fen state that will be used when random_state is set to False
    :param random_state: Boolean, Whether to use random states from the piece_list or use original_fen
    :param load_data: Boolean, Whether to load data from the previous training data file,
    if false it will overwrite the old file
    :param games_played: Int, Number of games that will be performed, default value 100.
    :return: None
    """
    data = []
    start = time.time()
    # if we want to load the old data
    if load_data:
        data = load_data_file()

    # counter for saving the progress to file, once it reaches 0, it saves the data
    checkpoint_counter = 50

    counter = 0
    # play games until enough games have been performed
    while counter < games_played:
        # generate a random fen notation if random_state is true
        if random_state:
            fen_str = generate_random_state(config.piece_list)
        else:
            fen_str = original_fen
        state = Chessboard(fen_str)
        game = TrainingGame(initial_state=state,
                            model=None)
        result = game.run()
        # if the result is either a win for white or a win for black, add it to the data
        # this is done so that we avoid draws in our training data.
        if result == Color.WHITE or result == Color.BLACK:
            data.append(game.get_history())
            counter += 1
            checkpoint_counter -= 1
            print(result)
            # every 50 games, we store the progress in the trainingdata file
            if checkpoint_counter == 0:
                checkpoint_counter = 50
                with bz2.BZ2File('trainingdata.bz2', 'w') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                print(f'saved data, games total:{len(data)}')


    # save the result one final time
    with bz2.BZ2File('trainingdata.bz2', 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print(len(data), ' | ', end-start, ' s')


def main():
    run_games("8/3r4/2kkk3/8/8/2KKK3/3R4/8 w - 0 1", random_state=True, games_played=10000)


def read_data():
    """Function that reads the data from the training data file that has been saved. useful for debugging.

    :return: None
    """
    with bz2.BZ2File('trainingdata.bz2', 'r') as f:
        data = pickle.load(f)
        for game in data:
            for (state, mcts, v) in game:
                print(state)
                print(v)

if __name__ == '__main__':
    #read_data()
    main()
