import time
import sys
sys.path.append('..')
import config
from TrainingGame import TrainingGame
from chess import Chessboard, Color
import pickle
import bz2
from state_generator import generate_random_state



def run_once(original_fen, random_state=False):
    data = []
    start = time.time()
    try:
        with bz2.BZ2File('trainingdata.bz2', 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        print(e)

    counter = 0
    while counter < 10:
        if random_state:
            fen_str = generate_random_state(config.piece_list)
        else:
            fen_str = original_fen
        state = Chessboard(fen_str)
        game = TrainingGame(initial_state=state,
                            model=None)
        result = game.run()
        if result != 'draw':
            data.append(game.get_history())
            counter += 1
            print(result)

    with bz2.BZ2File('trainingdata.bz2', 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print(len(data), ' | ', end-start, ' s')

def main():

    while True:
        run_once("k7/8/8/8/8/8/8/7R w - 0 1")


def read_data():

    with open('trainingdata.pkl', 'rb') as inp:
        data = pickle.load(inp)
        print(data)

if __name__ == '__main__':
    main()
