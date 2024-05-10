from Game.game import Game
from Model.model import Model
from Player.engine_player import EnginePlayer
from chess.chessboard import Chessboard
from chess.utils import Color


class Testing:
    """Helper class to play two models against each other"""

    def __init__(self, initial_state: str, model_1: Model, model_2: Model):
        self.initial_state = initial_state
        self.model_1 = model_1
        self.model_2 = model_2

    def test(self, games: int):
        """Method that runs a number of games between two models and returns the results

        :return: None
        """
        results_as_white = []
        for i in range(games - (games // 2)):
            player_1 = EnginePlayer(Chessboard(self.initial_state), self.model_1)
            player_2 = EnginePlayer(Chessboard(self.initial_state), self.model_2)
            game = Game(initial_state=Chessboard(self.initial_state), player_1=player_1, player_2=player_2)

            result = game.run()
            if result == Color.WHITE:
                results_as_white.append(1)
            elif result == Color.BLACK:
                results_as_white.append(0)
            else:
                results_as_white.append(0.5)

        results_as_black = []
        for i in range(games // 2):
            player_1 = EnginePlayer(Chessboard(self.initial_state), self.model_2)
            player_2 = EnginePlayer(Chessboard(self.initial_state), self.model_1)
            game = Game(initial_state=Chessboard(self.initial_state), player_1=player_1, player_2=player_2)

            result = game.run()
            if result == Color.WHITE:
                results_as_black.append(0)
            elif result == Color.BLACK:
                results_as_black.append(1)
            else:
                results_as_black.append(0.5)

        return print(f'Model 1 scored {sum(results_as_white) / len(results_as_white)} as white and {sum(results_as_black) / len(results_as_black)} as black.')
