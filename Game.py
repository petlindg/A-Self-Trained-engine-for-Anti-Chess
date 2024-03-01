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
        self.p1_tree = MCTS(root_state=self.current_state,
                            player=Color.WHITE)
        self.p2_tree = MCTS(root_state=self.current_state,
                            player=Color.BLACK)

    def game_ended(self):
        status = self.current_state.get_game_status()
        # if the game hasn't ended
        if status == 3:
            return False
        else:
            return True

# TODO fix the way that trees are handled

    def make_move(self, player: Color):
        if player == Color.WHITE:
            self.p1_tree.run()
            self.p1_tree.root_node.print_selectively(2)

            potential_nodes = self.p1_tree.root_node.children
            max_visits = 0
            best_move = None
            mcts_dist = []
            visit_total = 0
            for node in potential_nodes:
                visit_total += node.visits
                mcts_dist.append((node.visits, node.move))
                if node.visits > max_visits:
                    max_visits = node.visits
                    best_move = node.move
            # prepare the float distribution of all actions
            # so that the model can use it for backpropagation
            mcts_dist = [(n / visit_total, move) for (n, move) in mcts_dist]

            # add values to the game history recording all moves
            self.game_history.append((self.current_state, mcts_dist, self.current_state.player_to_move))
            self.current_state.move(best_move)
            self.p1_tree.update_tree(best_move, self.current_state)
            self.p2_tree.update_tree(best_move, self.current_state)
            print(f'player: {player}', best_move)
            return self.p1_tree.time_predicted

        else:
            self.p2_tree.run()
            self.p2_tree.root_node.print_selectively(2)

            potential_nodes = self.p2_tree.root_node.children
            max_visits = 0
            best_move = None
            mcts_dist = []
            visit_total = 0
            for node in potential_nodes:
                visit_total += node.visits
                mcts_dist.append((node.visits, node.move))
                if node.visits > max_visits:
                    max_visits = node.visits
                    best_move = node.move
            # prepare the float distribution of all actions
            # so that the model can use it for backpropagation
            mcts_dist = [(n / visit_total, move) for (n, move) in mcts_dist]

            # add values to the game history recording all moves
            self.game_history.append((self.current_state, mcts_dist, self.current_state.player_to_move))
            self.current_state.move(best_move)
            self.p1_tree.update_tree(best_move, self.current_state)
            self.p2_tree.update_tree(best_move, self.current_state)
            print(f'player: {player}', best_move)
            return self.p2_tree.time_predicted





    def run(self, model):
        self.p1_tree.model = model
        self.p2_tree.model = model
        start_time = time.time()
        predict_time = 0
        while not self.game_ended():
            print(self.current_state)
            print('player: ', self.current_state.player_to_move)
            time_predicted = self.make_move(player=self.current_state.player_to_move)
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


