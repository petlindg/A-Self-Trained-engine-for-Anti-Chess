import time

from Node import Node
from chess import Chessboard, Move, Color
from MCTS import MCTS
from keras.models import Model

from main import translate_moves_to_output


class TrainingGame:
    """
    A class representing one training game of antichess
    uses a single model, playing against itself in order to improve
    """

    def __init__(self, initial_state: Chessboard):
        self.current_state = initial_state
        self.game_over = False
        self.game_history = []
        self.swap = False

    def game_ended(self):
        status = self.current_state.get_game_status()
        # if the game hasn't ended
        if status == 3:
            return False
        else:
            return True



    def make_move(self, model: Model, player: Color, previous_tree: Node=None, ):
        tree = MCTS(root_state=self.current_state,
                    player=player,
                    model=model,
                    swap=self.swap,
                    old_tree_root=previous_tree)
        tree.run()
        potential_nodes = tree.root_node.children
        max_visits = 0
        best_move = None
        mcts_dist = []
        visit_total = 0
        for node in potential_nodes:
            visit_total += node.visits
            mcts_dist.append((node.visits, node.move))
            if node.visits > max_visits:
                max_visit = node.visits
                best_move = node.move
        # prepare the float distribution of all actions
        # so that the model can use it for backpropagation
        mcts_dist = [(n / visit_total, move) for (n, move) in mcts_dist]

        # add values to the game history recording all moves
        self.game_history.append((self.current_state, mcts_dist, self.current_state.player_to_move))
        self.current_state.move(best_move)
        # switch the swap variable
        self.swap = not self.swap
        return tree.select_child(best_move), tree.time_predicted
    def run(self, model):
        old_node = None
        start_time = time.time()
        predict_time = 0
        while not self.game_ended():
            print(self.current_state)
            old_node, time_predicted = self.make_move(model=model,
                                                      player=self.current_state.player_to_move,
                                                      previous_tree=old_node)
            predict_time += time_predicted
        end_time = time.time()
        total_time = end_time - start_time
        status = self.current_state.get_game_status()

        print('===============================')
        if status == 0:
            winner = Color.WHITE
            print('           white wins')
        elif status == 1:
            winner = Color.BLACK
            print('           black wins')
        else:
            winner = 'draw'
            print('             draw')
        print('===============================')
        print(f'Time taken: {total_time} | Time Predicted: {predict_time} | % {predict_time/total_time*100}')
        finalized_history = []
        for (state, mcts, player) in self.game_history:
            # if nobody won and it's a draw
            if winner == 'draw':
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0.5))
            # if winning player
            elif winner == player:
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 1))
            # if losing player
            elif winner is not player:
                finalized_history.append((state.translate_board(), translate_moves_to_output(mcts), 0))
        return finalized_history


class TestingGame:
    """
    a class representing one testing game of antichess,
    uses two different models playing against each other to evaluate
    their effectiveness against one another
    """


