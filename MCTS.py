import time
from math import sqrt
import random

# TODO add a function that returns possible states based on valid moves from this current state
# make this function also interact with the model by fetching the p values for these new states
# this function should return tuples in the form of ([(new_state, p)], v) for the expansion process
def possible_moves(state):
    # TESTING VALUES
    p1 = random.random()
    p2 = random.random()
    p3 = random.random()
    sumv = p1+p2+p3
    p1 = p1/sumv
    p2 = p2/sumv
    p3 = p3/sumv

    v = random.random()
    return [('left', p1),('middle', p2) , ('right', p3)], v

def ucb(node, c):
    return (node.p * c * sqrt(node.parent.visits)/(1+node.visits) + (node.value/node.visits if node.visits > 0 else 0))

# class defining the contents of a singular node
class Node:
    def __init__(self, p, state, parent_node, player, root_node):
        self.root_node = root_node
        self.parent = parent_node
        self.state = state
        self.children = []
        self.value = 0
        self.visits = 0
        self.v = 0
        self.p = p
        self.leaf = True
        # player is either 'self' or 'opponent', self == True, opponent == False
        # TODO figure out how this will interact with the model when it comes to evaluations
        # how the player that is currently trying to find the best move will interact with the model
        self.player = player

    # string method in order to represent the tree from the current node and down
    # when you try to print a particular node
    def __str__(self, level=0):
        string_buffer = []
        self.print_tree(string_buffer, "", "")
        return "".join(string_buffer)

    # method that can be called on a node which will print it with a given depth
    # any nodes that exist deeper than this depth will not be printed
    def print_selectively(self, depth):
        string_buffer = []
        self.print_tree(string_buffer, "", "", depth)
        print("".join(string_buffer))

    # method that will iteratively go through the tree and add on objects to the string_buffer
    # this buffer will then be used to print the entire tree in the terminal
    # depth parameter determines how deep the tree will print from the node, if depth=None then
    # the all children will be printed
    def print_tree(self, string_buffer, prefix, child_prefix, depth=None):
        if depth is None or depth > 0:
            string_buffer.append(prefix)
            p = round(self.p, 2)
            v = round(self.value, 2)
            V = round(self.v, 2)
            visits = self.visits

            info_text = f'(p:{p}|v:{v}|n:{visits}|V:{V})'
            string_buffer.append(info_text)
            string_buffer.append('\n')

            for i in range(0, len(self.children)):
                if i == len(self.children)-1:
                    self.children[i].print_tree(string_buffer, child_prefix + "└── ", child_prefix + "    ", (depth - 1 if depth else None))
                else:
                    self.children[i].print_tree(string_buffer, child_prefix + "├── ", child_prefix + "│   ", (depth - 1 if depth else None))


    # expand method which does both the expansion and the backpropagation, using backpropagate
    def expand(self):
        self.leaf = False
        # TODO implement possible_moves
        new_states, v = possible_moves(self.state)

        # the initial value v from the model is stored separate from the value variable
        # so that it can be used in backpropagation when training the actual model.
        self.v = v
        # creating the new child nodes, making sure to swap around the player variable
        for (new_state, p) in new_states:
            self.children.append(Node(p, new_state, self, not self.player, self.root_node))

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
        self.root_node = Node(1, root_state, None, True, None)
        self.root_node.root_node = self.root_node
        # number of MCTS iterations left, each iteration is one search
        self.iterations = iterations
        self.exploration_constant = sqrt(2)

    # method to perform a single search 'iteration'
    # goes through the 3 steps of an AlphaZero MCTS loop
    # selection, expansion, backpropagation,
    # backpropagation occurs inside the expand() function
    def search(self):
        leaf_node = self.selection(self.root_node)
        leaf_node.expand()

    # method that performs the selection process
    # goes down the tree based off of UCB until it hits a leaf node.
    def selection(self, current_node):
        while not current_node.leaf:
            current_node = max(current_node.children, key=lambda node: ucb(node, self.exploration_constant))

        return current_node

    # run method that will continually perform tree searches
    # until the iterations runs out
    def run(self):
        for i in range(0, self.iterations):
            self.search()

def main():
    #tree = MCTS('none', 11)
    #tree.run()
    performance()

def performance():
    iterations = 100
    start = time.time()
    tree = MCTS('none', iterations)
    tree.run()
    print(tree.root_node)

    end = time.time()
    print(f'runtime: {end-start} s | iterations per second: {iterations/(end-start)}')

# performance metrics: 100000 iterations takes about 3.6s on my (liam's) pc
# comparing that to my java implementation, which takes 0.272s on my pc
# the java implementation is roughly 13 times faster than the python one

if __name__ == "__main__":
    main()