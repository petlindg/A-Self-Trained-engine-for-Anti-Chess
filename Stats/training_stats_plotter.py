import pickle
from keras.callbacks import History
import matplotlib.pyplot as plt
import bz2
import json

class TrainingPlot:
    '''
    Class containing functionality to collect and plot training metrics
    '''
    data = {"loss": [], "accuracy": []}

    def load_data(self):
        '''
        Takes a json file from specific directory and loads the data into the class.

        :return: None
        '''
        with open("./training_metrics.json", "r") as f:
            self.data = json.load(f)
    
    def save_history_to_instance(self, history: History):
        metrics = history.history
        self.data["loss"].append(metrics["loss"][-1])
        self.data["accuracy"].append(metrics["accuracy"][-1])

    def save_history_to_json(self):
        with open("./training_metrics.json", "w") as f:
            json.dump(self.data, f)
    
    def plot_loss(self):
        plt.plot(self.data["loss"])
        plt.title("Change in Loss over Training Iterations")
        plt.xlabel("Training Iteration")
        plt.ylabel("Final Loss Value")
        plt.show()
    
    def plot_accuracy(self):
        plt.plot(self.data["accuracy"])
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