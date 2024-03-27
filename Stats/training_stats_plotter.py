import pickle
from keras.callbacks import History
import matplotlib.pyplot as plt


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
            with open("Stats/training_metrics.p", "rb") as f:
                while True:
                    try:
                        model: History = pickle.load(f)
                        self.process_loaded_data(model)
                    except EOFError:
                        break
        except FileNotFoundError:
            print("Training metrics data has not been saved yet")
    

    def store_pickled_data(model: History):
        with open("Stats/training_metrics.p", "ab") as f:
            pickle.dumps(model, f)
    
    def process_loaded_data(self, model_history: History):
        final_loss = model_history.history["loss"][-1]
        final_acc = model_history.history["accuracy"][-1]
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
