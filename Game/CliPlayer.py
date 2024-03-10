from chess import Chessboard
from chess import Move

class CliPlayer:
    """
    A class representing a player
    """

    def __init__(self, chessboard: Chessboard):
        self.state = chessboard

    def opponent_move(self, move: Move):
        self.state.move(move)

    def get_next_move(self):
        # Print the state to CLI
        print(self.state)

        # Get next move from player
        while True:
            next_move = input("Pls give next move")

            if self.state.move(next_move):
                return next_move
            else:
                print("Invalid move...")
