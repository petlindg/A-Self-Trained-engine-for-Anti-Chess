import bz2
import pickle

def main():
    data = []
    try:
        with bz2.BZ2File('trainingdata.bz2', 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        print(e)
    for game in data:
        for (state, p_list, winner) in game:
            print(state)
            print(p_list)
            print(winner)

if __name__=='__main__':
    main()