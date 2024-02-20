import unittest
import chess as cb

class TestMoveGen(unittest.TestCase):
    # class tests cardinality of movegeneration in different board states

    def test_pawn_white_takes(self):
        board = cb.Chessboard()
        board.init_board_test_pawn_white_takes()
        self.assertEqual(21, len(board.get_moves()))

    def test_pawn_white_moves(self):
        board = cb.Chessboard()
        board.init_board_test_pawn_white_moves()
        self.assertEqual(9, len(board.get_moves()))
    
    def test_knight_white_takes(self):
        board = cb.Chessboard()
        board.init_board_test_knight_white_takes()
        self.assertEqual(9, len(board.get_moves()))

    def test_knight_white_moves(self):
        board = cb.Chessboard()
        board.init_board_test_knight_white_moves()
        self.assertEqual(12, len(board.get_moves()))

    def test_bishop_white_takes(self):
        board = cb.Chessboard()
        board.init_board_test_bishop_white_takes()
        self.assertEqual(3, len(board.get_moves()))

    def test_bishop_white_moves(self):
        board = cb.Chessboard()
        board.init_board_test_bishop_white_moves()
        self.assertEqual(12, len(board.get_moves()))

    def test_rook_white_takes(self):
        board = cb.Chessboard()
        board.init_board_test_rook_white_takes()
        self.assertEqual(3, len(board.get_moves()))

    def test_rook_white_moves(self):
        board = cb.Chessboard()
        board.init_board_test_rook_white_moves()
        self.assertEqual(29, len(board.get_moves()))

    def test_queen_white_takes(self):
        board = cb.Chessboard()
        board.init_board_test_queen_white_takes()
        self.assertEqual(4, len(board.get_moves()))

    def test_queen_white_moves(self):
        board = cb.Chessboard()
        board.init_board_test_queen_white_moves()
        self.assertEqual(31, len(board.get_moves()))

    def test_king_white_takes(self):
        board = cb.Chessboard()
        board.init_board_test_king_white_takes()
        self.assertEqual(8, len(board.get_moves()))

    def test_king_white_moves(self):
        board = cb.Chessboard()
        board.init_board_test_king_white_moves()
        self.assertEqual(14, len(board.get_moves()))

    def test_pawn_black_takes(self):
        board = cb.Chessboard()
        board.init_board_test_pawn_black_takes()
        self.assertEqual(21, len(board.get_moves()))

    def test_pawn_black_moves(self):
        board = cb.Chessboard()
        board.init_board_test_pawn_black_moves()
        self.assertEqual(9, len(board.get_moves()))

    def test_knight_black_takes(self):
        board = cb.Chessboard()
        board.init_board_test_knight_black_takes()
        self.assertEqual(9, len(board.get_moves()))

    def test_knight_black_moves(self):
        board = cb.Chessboard()
        board.init_board_test_knight_black_moves()
        self.assertEqual(12, len(board.get_moves()))

    def test_bishop_black_takes(self):
        board = cb.Chessboard()
        board.init_board_test_bishop_black_takes()
        self.assertEqual(3, len(board.get_moves()))

    def test_bishop_black_moves(self):
        board = cb.Chessboard()
        board.init_board_test_bishop_black_moves()
        self.assertEqual(10, len(board.get_moves()))

    def test_rook_black_takes(self):
        board = cb.Chessboard()
        board.init_board_test_rook_black_takes()
        self.assertEqual(3, len(board.get_moves()))

    def test_rook_black_moves(self):
        board = cb.Chessboard()
        board.init_board_test_rook_black_moves()
        self.assertEqual(29, len(board.get_moves()))

    def test_queen_black_takes(self):
        board = cb.Chessboard()
        board.init_board_test_queen_black_takes()
        self.assertEqual(4, len(board.get_moves()))

    def test_queen_black_moves(self):
        board = cb.Chessboard()
        board.init_board_test_queen_black_moves()
        self.assertEqual(31, len(board.get_moves()))

    def test_king_black_takes(self):
        board = cb.Chessboard()
        board.init_board_test_king_black_takes()
        self.assertEqual(8, len(board.get_moves()))

    def test_king_black_moves(self):
        board = cb.Chessboard()
        board.init_board_test_king_black_moves()
        self.assertEqual(14, len(board.get_moves()))
    
    def test_enpassante_white(self):
        pass

    def test_en_passante_black(self):
        pass

def main():
    unittest.main()

if __name__ == "__main__":
    main()