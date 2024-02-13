class Chesspiece:
    '''
        Superclass for chesspieces in general, containing attributes that every chess piece has.
        It also contains getter methods that help accessing data about a given piece.
    '''
    def __init__(self, color, position):
        self.color = color
        self.position = position

    def get_color(self):
        return self.color
    
    def get_position(self):
        return self.position

class Pawn(Chesspiece):
    
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
    
    def select_move(self):

        move = str(input('What move would you like to make? '))

        if move in self.get_captures() + self.get_pawn_moves():
            print(move)

        else:
            print('Invalid move')
            self.select_move()


    
    
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
    
p = Pawn()

print()