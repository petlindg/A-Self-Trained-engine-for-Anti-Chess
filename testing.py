from copy import deepcopy

from Game.TrainingGame import TrainingGame
from chess import Color


class Testing:
    """Helper class to play two models against each other"""

    def __init__(self, initial_state, model_1, model_2):
        self.initial_state = initial_state
        self.model_1 = model_1
        self.model_2 = model_2

    def test(self, games: int):
        """Method that runs a number of games between two models and returns the results

        :return: None
        """

        results_as_white = []
        for i in range(games - (games // 2)):
            game = TrainingGame(initial_state=deepcopy(self.initial_state), white_model=self.model_1, black_model=self.model_2)

            result = game.run()

            if result == Color.WHITE:
                results_as_white.append(1)
            elif result == Color.BLACK:
                results_as_white.append(0)
            else:
                results_as_white.append(0.5)

        results_as_black = []
        for i in range(games // 2):
            game = TrainingGame(initial_state=deepcopy(self.initial_state), white_model=self.model_2, black_model=self.model_1)
            result = game.run()

            if result == Color.WHITE:
                results_as_black.append(0)
            elif result == Color.BLACK:
                results_as_black.append(1)
            else:
                results_as_black.append(0.5)

        return print(f'Model 1 scored {sum(results_as_white) / len(results_as_white)} as white and {sum(results_as_black) / len(results_as_black)} as black.')
