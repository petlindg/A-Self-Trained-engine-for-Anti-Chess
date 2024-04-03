import pickle
from keras.callbacks import History
import matplotlib.pyplot as plt
import bz2

class TrainingPlot:
    '''
    Class containing functionality to collect and plot training metrics
    '''
    data = [[],[]]

    def load_data(self):
        '''
        Takes a pickle file from specific directory and unpickles/loads the data into the class.

        :return: None
        '''
        try:
            with bz2.BZ2File("./training_metrics.bz2", "rb") as f:
                while True:
                    try:
                        model: dict = pickle.load(f)
                        self.process_loaded_data(model)
                    except EOFError:
                        break
        except FileNotFoundError:
            print("Training metrics data has not been saved yet")
    
    # def compress(self):
    #     with open("./training_metrics.p", "rb") as f:
    #         pickled = pickle.load(f)
    #         compressed = bz2.compress(pickled)
    #         with bz2.open("./training_metrics.bz2", "wb") as f:
    #             f.write(compressed)

    def pickle_history(self, history: dict):
        with open("./training_metrics.p", "ab") as f:
            pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    
    def process_loaded_data(self, model_history: dict):
        final_loss = model_history["loss"][-1]
        final_acc = model_history["accuracy"][-1]
        self.data[0].append(final_loss)
        self.data[1].append(final_acc)

    def plot_loss(self):
        plt.plot(self.data[0])
        plt.title("Change in Loss over Training Iterations")
        plt.xlabel("Training Iteration")
        plt.ylabel("Final Loss Value")
        plt.show()
    
    def plot_accuracy(self):
        plt.plot(self.data[1])
        plt.title("Change in Accuracy over Training Iterations")
        plt.xlabel("Training Iteration")
        plt.ylabel("Final Accuracy Value")
        plt.show()


    def show_statistics(self):
        self.load_data()
        self.plot_loss()
        self.plot_accuracy()

if __name__ == '__main__':
    train = TrainingPlot()
    train.show_statistics()