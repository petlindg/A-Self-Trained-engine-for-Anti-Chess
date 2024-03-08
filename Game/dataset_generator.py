from TrainingGame import TrainingGame
from chess import Chessboard
import pickle


def main():

    final_data = []
    for i in range(0, 100):
        state = Chessboard("k7/8/8/8/8/8/8/7R w - 0 1")
        game = TrainingGame(initial_state=state,
                            model=None)
        game.run()
        final_data.append(game.get_history())

    with open('trainingdata.pkl', 'wb') as outp:
        pickle.dump(final_data, outp, pickle.HIGHEST_PROTOCOL)

def read_data():

    with open('trainingdata.pkl', 'rb') as inp:
        data = pickle.load(inp)
        print(data)
if __name__ == '__main__':
    main()
    read_data()