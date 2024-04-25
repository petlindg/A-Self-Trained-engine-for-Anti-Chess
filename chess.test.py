import unittest
import chess as cb

class TestMoveGen(unittest.TestCase):
    # class tests cardinality of generated legal moves in different board states

    def test_pawn_white_takes(self):
        board = cb.Chessboard("kk4kk/PP1P2PP/8/8/8/4k3/3P2P1/8 w - 0 1")
        self.assertEqual(21, len(board.get_moves()))

    def test_pawn_white_moves(self):
        board = cb.Chessboard("8/6P1/8/8/8/k7/PkP4P/8 w - 0 1")
        self.assertEqual(9, len(board.get_moves()))
    
    def test_knight_white_takes(self):
        board = cb.Chessboard("8/8/8/5N1k/NNN1N3/3N2k1/1k2N3/3N1N1N w - 0 1")
        self.assertEqual(9, len(board.get_moves()))

    def test_knight_white_moves(self):
        board = cb.Chessboard("N7/1N6/8/8/8/k7/6N1/7N w - 0 1")
        self.assertEqual(12, len(board.get_moves()))

    def test_bishop_white_takes(self):
        board = cb.Chessboard("8/8/8/4k2k/3k4/5K2/1B6/2kk3B w - 0 1")
        self.assertEqual(2, len(board.get_moves()))

    def test_bishop_white_moves(self):
        board = cb.Chessboard("8/8/2k5/8/5B2/5P2/8/7B w - 0 1")
        self.assertEqual(12, len(board.get_moves()))

    def test_rook_white_takes(self):
        board = cb.Chessboard("8/8/7k/2R4k/8/8/8/2k1K2R w - 0 1")
        self.assertEqual(3, len(board.get_moves()))

    def test_rook_white_moves(self):
        board = cb.Chessboard("8/8/8/8/8/7R/4R3/2k1P2R w - 0 1")
        self.assertEqual(29, len(board.get_moves()))

    def test_queen_white_takes(self):
        board = cb.Chessboard("8/1Q6/1K2k3/8/1kk1Q1k1/8/4k3/8 w - 0 1")
        self.assertEqual(4, len(board.get_moves()))

    def test_queen_white_moves(self):
        board = cb.Chessboard("8/8/8/3k4/5Q2/5P2/5P2/3k1P1Q w - 0 1")
        self.assertEqual(31, len(board.get_moves()))

    def test_king_white_takes(self):
        board = cb.Chessboard("8/8/8/8/k7/K4k1k/5kKk/K4kkk w - 0 1")
        self.assertEqual(8, len(board.get_moves()))

    def test_king_white_moves(self):
        board = cb.Chessboard("K7/8/8/8/3K4/8/8/k6K w - 0 1")
        self.assertEqual(14, len(board.get_moves()))

    def test_pawn_black_takes(self):
        board = cb.Chessboard("8/3p2p1/4K3/8/8/8/pp1p2pp/KK4KK b - 0 1")
        self.assertEqual(21, len(board.get_moves()))

    def test_pawn_black_moves(self):
        board = cb.Chessboard("8/pKp4p/K7/8/8/8/6p1/8 b - 0 1")
        self.assertEqual(9, len(board.get_moves()))

    def test_knight_black_takes(self):
        board = cb.Chessboard("8/8/8/5n1K/nnn1n3/3n2K1/1K2n3/3n1n1n b - 0 1")
        self.assertEqual(9, len(board.get_moves()))

    def test_knight_black_moves(self):
        board = cb.Chessboard("n7/1n6/8/8/8/K7/6n1/7n b - 0 1")
        self.assertEqual(12, len(board.get_moves()))

    def test_bishop_black_takes(self):
        board = cb.Chessboard("8/8/8/4K2K/3K4/5k2/1b6/2KK3b b - 0 1")
        self.assertEqual(2, len(board.get_moves()))

    def test_bishop_black_moves(self):
        board = cb.Chessboard("8/8/2K5/8/8/5p2/5b2/7b b - 0 1")
        self.assertEqual(10, len(board.get_moves()))

    def test_rook_black_takes(self):
        board = cb.Chessboard("8/8/7K/2r4K/8/8/8/2K1k2r b - 0 1")
        self.assertEqual(3, len(board.get_moves()))

    def test_rook_black_moves(self):
        board = cb.Chessboard("8/8/8/8/8/7r/4r3/2K1p2r b - 0 1")
        self.assertEqual(29, len(board.get_moves()))

    def test_queen_black_takes(self):
        board = cb.Chessboard("8/1q6/1k2K3/8/1KK1q1K1/8/4K3/8 b - 0 1")
        self.assertEqual(4, len(board.get_moves()))

    def test_queen_black_moves(self):
        board = cb.Chessboard("8/8/8/3K4/5q2/5p2/5p2/3K1p1q b - 0 1")
        self.assertEqual(31, len(board.get_moves()))

    def test_king_black_takes(self):
        board = cb.Chessboard("8/8/8/8/K7/k4K1K/5KkK/k4KKK b - 0 1")
        self.assertEqual(8, len(board.get_moves()))

    def test_king_black_moves(self):
        board = cb.Chessboard("k7/8/8/8/3k4/8/8/K6k b - 0 1")
        self.assertEqual(14, len(board.get_moves()))
    
    def test_enpassante_white(self):
        board = cb.Chessboard("8/8/8/pPpPP3/8/8/8/8 w C6 0 1")
        self.assertEqual(2, len(board.get_moves()))

    def test_en_passante_black(self):
        board = cb.Chessboard("8/8/8/8/PpPpp3/8/8/8 b C3 0 1")
        self.assertEqual(2, len(board.get_moves()))

    def test_move_valid_1(self):
        board = cb.Chessboard()
        m = cb.Move(cb.u64(11), cb.u64(27))
        self.assertTrue(board.move(m))

    def test_move_valid_2(self):
        board = cb.Chessboard()
        m = cb.Move(cb.u64(11), cb.u64(19))
        self.assertTrue(board.move(m))
    
    def test_move_invalid_1(self):
        board = cb.Chessboard()
        m = cb.Move(cb.u64(11), cb.u64(26))
        self.assertFalse(board.move(m))
    
    def test_move_invalid_2(self):
        board = cb.Chessboard()
        m = cb.Move(cb.u64(11), cb.u64(34))
        self.assertFalse(board.move(m))

    def test_enpassant_unmove(self):
        fen = "8/8/8/8/1p6/8/P7/8 w - 0 1"
        moves_alg = "a2a4 b4a3 b4a3"
        board = cb.Chessboard(fen)
        moves = [cb.algebraic_to_move(move_alg) for move_alg in moves_alg.split()]

        self.assertTrue(board.move(moves.pop(0)))
        self.assertTrue(board.move(moves.pop(0)))
        board.unmove()
        self.assertTrue(board.move(moves.pop(0)))

class TestGameState(unittest.TestCase):
    # tests that the get_game_status() function works correctly
    def test_stalemate_white(self):
        board = cb.Chessboard("8/8/3p4/3Pp3/4P3/8/8/8 w - 0 1")
        self.assertEqual(0, board.get_game_status())

    def test_stalemate_black(self):
        board = cb.Chessboard("8/8/3p4/3Pp3/4P3/8/8/8 b - 0 1")
        self.assertEqual(1, board.get_game_status())

    def test_draw_repetition(self):
        board = cb.Chessboard("k7/8/8/8/8/8/8/7K w - 0 1")
        w_m1 = cb.Move(cb.u64(0), cb.u64(1))
        b_m1 = cb.Move(cb.u64(63), cb.u64(62))
        w_m2 = cb.Move(cb.u64(1), cb.u64(0))
        b_m2 = cb.Move(cb.u64(62), cb.u64(63))
        board.move(w_m1)
        self.assertFalse(board.is_draw())
        board.move(b_m1)
        self.assertFalse(board.is_draw())
        board.move(w_m2)
        self.assertFalse(board.is_draw())
        board.move(b_m2)
        self.assertFalse(board.is_draw())
        board.move(w_m1)
        self.assertFalse(board.is_draw())
        board.move(b_m1)
        self.assertFalse(board.is_draw())
        board.move(w_m2)
        self.assertFalse(board.is_draw())
        board.move(b_m2)
        self.assertTrue(board.is_draw())

    def test_draw_no_progress(self):
        board = cb.Chessboard("k7/8/8/8/8/8/8/7K w - 0 1")
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
        board.move(b_m1)
        self.assertTrue(board.is_draw())
        self.assertTrue(board._check_no_progress())
            
class TestTranslations(unittest.TestCase):
    def test_fen_translation1(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1"
        board = cb.Chessboard(fen)
        new_fen = board.get_fen()
        self.assertEqual(fen, new_fen)

    def test_fen_translation2(self):
        fen = "8/8/8/8/PpPpp3/8/8/8 b C3 3 5"
        board = cb.Chessboard(fen)
        new_fen = board.get_fen()
        self.assertEqual(fen, new_fen)

    def eq_rep(self, rep1, rep2):
        """
        Helper function to assert translate_board() functionality
        """
        if len(rep1) != len(rep2):
            raise RuntimeError("Wrong length in input representations")
        for x in range(8):
            for y in range(8):
                for z in range(17):
                    if rep1[0][x][y][z] != rep2[0][x][y][z]:
                        return False
        return True
    
    def test_translate_board_after_unmove_1(self):
        moves_alg = "e2e4 a7a6 f1a6"
        board = cb.Chessboard()
        moves = [cb.algebraic_to_move(move_alg) for move_alg in moves_alg.split()]

        rep1 = board.translate_board()
        for move in moves:
            board.move(move)
        for _ in range(3):
            board.unmove()
        rep2 = board.translate_board()
        
        self.assertTrue(self.eq_rep(rep1, rep2))

    def test_translate_board_after_unmove_2(self):
        fen = "k7/8/8/8/8/8/8/7R w - 0 1"
        moves_alg = "h1c1 a8a7 c1c5 a7a8 c5c6 a8b8 c6c7 b8c7"
        board = cb.Chessboard(fen)
        moves = [cb.algebraic_to_move(move_alg) for move_alg in moves_alg.split()]

        rep_list = []

        for move in moves:
            rep_list.append(board.translate_board())
            board.move(move)
        
        for _ in moves:
            board.unmove()
            self.assertTrue(self.eq_rep(rep_list.pop(), board.translate_board()))
 
 
def main():
    unittest.main()


if __name__ == "__main__":
    main()