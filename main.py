from MCTS import MCTS
from collections import deque




# a class representing the generation of the Training Data
class TrainingData:
    def __init__(self, max_buffer_size, initial_state ,tree_iterations):
        # a buffer with a max size operating on a first in first out principle
        # during the training process, the oldest data will be replaced with the newest data
        # inside this buffer
        self.buffer = deque(max_buffer_size)
        self.training = True
        self.initial_state = initial_state
        self.tree_iterations = tree_iterations

    # method that will continually generate training data through self play while self.training is true
    def train(self):
        while self.training:
            tree = MCTS(self.initial_state, self.tree_iterations)
            tree.run()
            tree_distribution = tree.root_node.children

class Game:
    def __init__(self, initial_state):
        self.current_player = 'white'
        self.current_state = initial_state
        self.game_over = False

def main():
    pass

if __name__ == "__main__":
    main()