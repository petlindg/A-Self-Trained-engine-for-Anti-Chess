import time
import sys
sys.path.append('..')
import config
from TrainingGame import TrainingGame
from chess import Chessboard, Color
import pickle
import bz2
from state_generator import generate_random_state
from multiprocessing import Pool, cpu_count, Semaphore, Process, Manager


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


def run_multi_games(original_fen, random_state=False, load_data=False):
    """Function to run games for the dataset_generator, by utilizing parallellization.

    :param original_fen:
    :param random_state:
    :param load_data:
    :return:
    """
    # lock for when a process wants to access the trainingdata.bz2 file
    file_lock = Semaphore(1)
    start_time = time.time()
    list_processes = []
    # start processes in accordance to the number of logical cpu cores.
    for i in range(cpu_count()):
        print(i)
        p = Process(target=single_play_process, args=(file_lock,
                             random_state, original_fen, start_time,))
        p.daemon = True
        p.start()
        list_processes.append(p)

    # keep alive loop (could be made more useful in the future)
    # once the main process ends, the child processes also end due to daemon = True
    while True:
        time.sleep(2)




def single_play_process(file_lock, random_state, original_fen, start_time):
    """Function that is used to run a single process, will continually run until the parent process
    ends.

    :param file_lock: Semaphore, the Semaphore for the file trainingdata.bz2
    :param random_state: Boolean, whether to generate a random state or not
    :param original_fen: String, original fen notation string, will be used if random_state = False
    :param start_time: Time, starting time of the process, used for debugging purposes and performance metrics
    :return: None
    """
    # how often the process will acquire the lock and save the newly generated games
    checkpoint_games = 2
    # process internal list of newly made games
    internal_data = []
    while True:
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
            print(result)
            internal_data.append(game.get_history())
            games = len(internal_data)
            # if we have reached the checkpoint, acquire lock and save to the file
            # by appending the old games with the new.
            if games % checkpoint_games == 0:

                file_lock.acquire()
                total_data = load_data_file()
                total_data = total_data + internal_data
                with bz2.BZ2File('trainingdata.bz2', 'w') as f:
                    pickle.dump(total_data, f, pickle.HIGHEST_PROTOCOL)
                print(f'saved data, games total:{len(total_data)}')
                # empty the internal data list
                internal_data = []
                # release the lock for other processes.
                file_lock.release()


    # rough performance tests:
    # 16 processes, 10 games in 37s,
    # 8 processes 10 games in 70s
    # 1 process, 10 games 464s
    # result: ~12.5 times faster with 16 times the processes
    # amd ryzen 5800X3D 16 cores 100 games 3kr vs 3KR: 9 minutes 30s

def main():
    #run_games("8/3r4/2kkk3/8/8/2KKK3/3R4/8 w - 0 1", random_state=False, games_played=10000)
    run_multi_games("8/3r4/2kkk3/8/8/2KKK3/3R4/8 w - 0 1", random_state=False)

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
