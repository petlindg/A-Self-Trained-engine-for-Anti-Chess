from math import sqrt


# TODO add a function that returns possible states based on valid moves from this current state
# TODO make this function also interact with the model by fetching the p values for these new states
# TODO this function should return tuples in the form of (new_state, p) for the expansion process
def possible_moves(state):
    pass


def ucb(node, c):
    return node.value/node.visits + node.p * c * sqrt(node.parent.visits)/(1+node.visits)

# class defining the contents of a singular node
class Node:
    def __init__(self, p, state, parent_node):
        self.parent = parent_node
        self.state = state
        self.children = []
        self.value = 0
        self.visits = 0
        self.p = p
        self.leaf = True

    def expand(self):
        self.leaf = False
        # TODO implement possible_moves
        new_states = possible_moves(self.state)
        for (new_state, p) in new_states:
            self.children.append(Node(p, new_state, self))


# class defining a singular MCTS search
class MCTS:
    def __init__(self, root_state, iterations):
        # defining the root node of the tree
        self.root_node = Node(1, root_state, None)
        # number of MCTS iterations left, each iteration is one search
        self.iterations_left = iterations
        self.exploration_constant = sqrt(2)

    # method to perform a single search 'iteration'
    # goes through the 3 steps of an AlphaZero MCTS loop
    # selection, expansion, backpropagation
    def search(self):
        leaf_node = self.selection(self.root_node)
        leaf_node.expand()

    def selection(self, current_node):
        while not current_node.leaf:
            ucb_set = {}
            for child in current_node.children:
                ucb_set[child] = ucb(child, self.exploration_constant)
            current_node = max(ucb_set, key=ucb_set.get)
        return current_node

    # UCB formula for returning the ucb score of a particular node

    # performs searches until sufficiently sized tree
    def run(self):
        while self.iterations_left > 0:
            self.search()


def main():
    pass


if __name__ == "__main__":
    main()