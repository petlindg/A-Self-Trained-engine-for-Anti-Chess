from MCTS import MCTS
from collections import deque


# a class representing the generation of the Training Data
class TrainingData:
    def __init__(self, max_buffer_size, initial_state, tree_iterations):
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
            pass


# class representing an entire game session
class Game:
    def __init__(self, initial_state, tree_iterations):
        self.current_player = 'white'
        self.current_state = initial_state
        self.game_over = False
        self.game_history = []
        self.tree_iterations = tree_iterations

    # method to check if the game is over
    # and if so, do the required actions to finalize the game
    def check_end_state(self):
        pass

    # predict the best move for the current player and make said move
    def make_move(self):
        tree = MCTS(self.current_state, self.tree_iterations)
        tree.run()
        potential_nodes = tree.root_node.children
        max_visit = 0
        best_state = None
        # iterate through all the possible children to the current state
        # and select the one with the highest visit count, this is the move to perform
        mcts_dist = []
        visit_total = 0
        for node in potential_nodes:
            visit_total += node.visits
            mcts_dist.append(node.visits)
            if node.visits > max_visit:
                max_visit = node.visits
                best_state = node.state
        # prepare the float distribution of all actions
        # TODO this should match up with the output from the model
        # so that the model can use it for backpropagation
        mcts_dist = [n/visit_total for n in mcts_dist]

        # add values to the game history recording all moves
        self.game_history.append((self.current_state, mcts_dist, tree.root_node.v, self.current_player))

        # update the current node
        self.current_state = best_state


def main():
    pass

if __name__ == "__main__":
    main()