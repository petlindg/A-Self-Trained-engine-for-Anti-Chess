from chess import Chessboard, algebraic_to_move, move_to_algebraic


class CliPlayer:
    """
    A class representing a player
    """

    def __init__(self, chessboard):
        self.chessboard = chessboard

    def get_move(self):
        # Print the state to CLI
        print(self.chessboard)

        # Get next move from player
        while True:
            next_move = algebraic_to_move(input("Input next move: "))
            print("\n")

            if True: # chessboard.is_valid(next_move):
                return next_move
            else:
                print(move_to_algebraic(next_move) + " is not a valid move")

    def update(self, move):
        self.chessboard.move(move)
