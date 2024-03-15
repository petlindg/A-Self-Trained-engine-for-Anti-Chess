import pickle

def main():
    fp = "trainingData/training_data_2024-03-13T13"
    file = open(fp, 'rb')
    game_history = pickle.load(file)
    for (state, p_list, winner) in game_history:
        print(state)
        print(p_list)
        print(winner)

if __name__=='__main__':
    main()