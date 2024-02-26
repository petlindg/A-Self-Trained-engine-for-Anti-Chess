import unittest
import chess as cb

class TestMoveGen(unittest.TestCase):
    # class tests cardinality of generated legal moves in different board states

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
        self.assertEqual(2, len(board.get_moves()))

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
        self.assertEqual(2, len(board.get_moves()))

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

class TestGameState(unittest.TestCase):
    # tests that the get_game_status() function works correctly
    def test_stalemate_white(self):
        board = cb.Chessboard()
        board.init_board_test_stalemate_white()
        self.assertEqual(0, board.get_game_status())

    def test_stalemate_black(self):
        board = cb.Chessboard()
        board.init_board_test_stalemate_black()
        self.assertEqual(1, board.get_game_status())

    def test_draw_repetition(self):
        board = cb.Chessboard()
        board.init_board_test_draw_repetition()
        w_m1 = cb.Move(cb.u64(0), cb.u64(1))
        b_m1 = cb.Move(cb.u64(63), cb.u64(62))
        w_m2 = cb.Move(cb.u64(1), cb.u64(0))
        b_m2 = cb.Move(cb.u64(62), cb.u64(63))
        self.assertFalse(board.move(w_m1))
        self.assertFalse(board.move(b_m1))
        self.assertFalse(board.move(w_m2))
        self.assertFalse(board.move(b_m2))
        self.assertFalse(board.move(w_m1))
        self.assertFalse(board.move(b_m1))
        self.assertFalse(board.move(w_m2))
        self.assertTrue(board.move(b_m2))

    def test_draw_no_progress(self):
        board = cb.Chessboard()
        board.init_board_test_draw_no_progress()
        w_m1 = cb.Move(cb.u64(0), cb.u64(1))
        b_m1 = cb.Move(cb.u64(63), cb.u64(62))
        w_m2 = cb.Move(cb.u64(1), cb.u64(0))
        b_m2 = cb.Move(cb.u64(62), cb.u64(63))
        for _ in range(12):
            board.move(w_m1)
            self.assertFalse(board._check_no_progress())
            board.move(b_m1)
            self.assertFalse(board._check_no_progress())
            board.move(w_m2)
            self.assertFalse(board._check_no_progress())
            board.move(b_m2)
            self.assertFalse(board._check_no_progress())
        board.move(w_m1)
        self.assertFalse(board._check_no_progress())
        self.assertTrue(board.move(b_m1))
        self.assertTrue(board._check_no_progress())
            
 
def main():
    unittest.main()


if __name__ == "__main__":
    main()