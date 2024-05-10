from chess.move import Move


class Player:
    def get_next_move(self):
        """Request a move from the player."""
        pass

    def update_tree(self, move: Move):
        """Update the player's board state."""
        pass

    def get_time_predicted(self):
        """Get the time spent thinking."""
        return 0