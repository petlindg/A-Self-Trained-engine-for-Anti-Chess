import time
from math import sqrt
import random

# TODO add a function that returns possible states based on valid moves from this current state
# make this function also interact with the model by fetching the p values for these new states
# this function should return tuples in the form of ([(new_state, p)], v) for the expansion process
def possible_moves(state):
    # TESTING VALUES
    p1 = random.random()
    p2 = (1-p1)
    v = random.random()
    return [('left', p1), ('right', p2)], v


def ucb(node, c):
    try:
        return node.value/node.visits + node.p * c * sqrt(node.parent.visits)/(1+node.visits)
    except ZeroDivisionError: # if node.visits = 0 ignore the first half and simplify second half of the formula
        return node.p * c * sqrt(node.parent.visits)

# class defining the contents of a singular node
class Node:
    def __init__(self, p, state, parent_node, player):
        self.parent = parent_node
        self.state = state
        self.children = []
        self.value = 0
        self.visits = 0
        self.p = p
        self.leaf = True
        # player is either 'self' or 'opponent', self == True, opponent == False
        self.player = player

    # string method in order to represent the tree from the current node and down
    def __str__(self, level=0):
        s = "\t"*level + f'(s:{self.state} v:{round(self.value, 2)}, ' \
                         f'n:{self.visits}, p:{round(self.p, 2)}, Pl: {self.player})'+'\n'
        for child in self.children:
            s += child.__str__(level+2)
        return s

    # expand method which does both the expansion and the backpropagation, using backpropagate
    def expand(self):
        self.leaf = False
        # TODO implement possible_moves
        new_states, v = possible_moves(self.state)

        # creating the new child nodes, making sure to swap around the player variable
        for (new_state, p) in new_states:
            self.children.append(Node(p, new_state, self, not self.player))

        # backpropagation process
        self.backpropagate(self, v, self.player)

    # backpropagation function, node is the current node to backpropagate to
    # v is the result value from the model and player is the player that performed the move
    # that resulted in the v value.
    def backpropagate(self, node, v, player):

        # if the actor performing the action with result v is the same as the current node
        # then we increase the value for the node by v
        if node.player == player:
            node.value += v
        # if the actor performing the action with result v is the opposite of the current node
        # then increase the value by (1-v) to get the oppositions value
        else:
            node.value += (1-v)

        node.visits += 1
        # if the parent is not none we aren't at the root yet and should continue
        if node.parent is not None:
            self.backpropagate(node.parent, v, player)


# class defining a singular MCTS search
class MCTS:
    def __init__(self, root_state, iterations):
        # defining the root node of the tree
        self.root_node = Node(1, root_state, None, True)
        # number of MCTS iterations left, each iteration is one search
        self.iterations_left = iterations
        self.exploration_constant = sqrt(2)

    # method to perform a single search 'iteration'
    # goes through the 3 steps of an AlphaZero MCTS loop
    # selection, expansion, backpropagation,
    # backpropagation occurs inside the expand() function
    def search(self):
        leaf_node = self.selection(self.root_node)
        leaf_node.expand()
        #print(self.root_node)

    # method that performs the selection process
    # goes down the tree based off of UCB until it hits a leaf node.
    def selection(self, current_node):
        while not current_node.leaf:
            ucb_set = {}
            for child in current_node.children:
                ucb_set[child] = ucb(child, self.exploration_constant)
            current_node = max(ucb_set, key=ucb_set.get)
        return current_node

    # run method that will continually perform tree searches
    # until the iterations runs out
    def run(self):
        while self.iterations_left > 0:
            self.search()
            self.iterations_left -= 1


def main():
    #tree = MCTS('none', 11)
    #tree.run()
    performance()

def performance():
    iterations = 100000
    start = time.time()
    tree = MCTS('none', iterations)
    tree.run()
    end = time.time()
    print(f'runtime: {end-start} s | iterations per second: {iterations/(end-start)}')

# performance metrics: 100000 iterations takes about 3.6s on my (liam's) pc
# comparing that to my java implementation, which takes 0.272s on my pc
# the java implementation is roughly 13 times faster than the python one

if __name__ == "__main__":
    main()