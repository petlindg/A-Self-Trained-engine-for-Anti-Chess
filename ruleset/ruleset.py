class Chesspiece:

    def __init__(self, color, position):
        self.color = color
        self.position = position

class Pawn(Chesspiece):
    pass
    def __init__(self, first_move = True):
        super.__init__()
        self.first_move = first_move
    
    def get_pawn_moves(self, startsquare, endsquare, promotion = False):

        legal_moves = []

        # add logic to moves

        return legal_moves

    def get_captures(self, startsquare, endsquare, promotion = False, en_passant = False):

        legal_captures = []

        # add logic to captures

        return legal_captures
    
    
class Knight(Chesspiece):
    pass

class Bishop(Chesspiece):
    pass

class Rook(Chesspiece):
    pass

class Queen(Chesspiece):
    pass

class King(Chesspiece):
    pass
    
