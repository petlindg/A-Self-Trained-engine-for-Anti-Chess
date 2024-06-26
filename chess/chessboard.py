from numpy import uint64 as u64
from numpy import uint8 as u8
from numpy import left_shift as ls
from numpy import right_shift as rs
from numpy import bitwise_or as b_or
from numpy import bitwise_and as b_and
from numpy import bitwise_not as b_not
from numpy import bitwise_xor as b_xor
from numpy import zeros, ndarray, uint, array
from numpy import any, all
from typing import List
import sys
sys.path.append('..')

from chess.utils import Color, Piece, alg_sq_to_index, print_bb
from chess.move import Move
import chess.lookup as lookup


class Chessboard():
    # main bitboard variable for representing the 12 bitboards.
    # The first index represents color, 0 = white, 1 = black
    # The second index represents piece-type.
    # 0 = pawns, 1 = knights, 2 = bishops, 3 = rooks, 4 = queens, 5 = kings
    bitboards : ndarray
    # index 0 = bitboard of combined white pieces
    # index 1 = bitboard of combined black pieces
    # index 2 = bitboard of combined black and white pieces
    combined : ndarray
    # bitboard to keep track of en-passant-able squares
    enpassante : u64
    # other logic
    player_to_move : Color
    not_player_to_move : Color
    no_progress_counter : List[u8]
    # a list of bitboards that represent the state
    repetitions_list : List[ndarray]
    # a bitboard to keep track of combined pawn positions
    pawns : u64
    # keeping track of number of pieces on the board
    piece_count : u8
    # a list of moves to hold in memory for avoiding generating the same moves twice
    moves : List[Move]
    # the normal chess move counter, updates when black moves
    move_counter : u8
    

    # init empty board
    def __init__(self, fen:str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1"):
        """
        Initializes the chessboard. By default with standard chess configuration
        alternatively by a string given in FEN format, excluding the field for castling.

        :param fen: String stating boardstate in FEN format, excluding the castling rights field
        """
        self.bitboards = zeros((2, 6), dtype=u64)
        self.combined  = zeros(3, dtype=u64)
        self.repetitions_list = []
        self.no_progress_counter = []
        self._init_fen(fen)
        self.pawns = self._get_pawns()
        self.piece_count = self._get_piece_count()
        self.enpassant_list = []
        self.enpassant_list.append(self.enpassante)
        self.moves = []
        
    def __str__(self):
        str_builder = ['a b c d e f g h \n']
        mask_bb = u64(pow(2, 63))
        n = 8
        for i in range(64):
            if not i % 8:
                if i != 0:
                    str_builder.append(f'{n}\n')
                    n = n - 1
            # if white pawn
            if b_and(self.bitboards[0][0], mask_bb):
                str_builder.append('P ')
            # if black pawn
            elif b_and(self.bitboards[1][0], mask_bb):
                str_builder.append('p ')

            # if white knight
            elif b_and(self.bitboards[0][1], mask_bb):
                str_builder.append('N ')
            # if black knight
            elif b_and(self.bitboards[1][1], mask_bb):
                str_builder.append('n ')

            # if white bishop
            elif b_and(self.bitboards[0][2], mask_bb):
                str_builder.append('B ')
            # if black bishop
            elif b_and(self.bitboards[1][2], mask_bb):
                str_builder.append('b ')

            # if white rook
            elif b_and(self.bitboards[0][3], mask_bb):
                str_builder.append('R ')
            # if black rook
            elif b_and(self.bitboards[1][3], mask_bb):
                str_builder.append('r ')

            # if white queen
            elif b_and(self.bitboards[0][4], mask_bb):
                str_builder.append('Q ')
            # if black queen
            elif b_and(self.bitboards[1][4], mask_bb):
                str_builder.append('q ')

            # if white king
            elif b_and(self.bitboards[0][5], mask_bb):
                str_builder.append('K ')
            # if black king
            elif b_and(self.bitboards[1][5], mask_bb):
                str_builder.append('k ')
            else:
                str_builder.append('. ')
            mask_bb = rs(mask_bb, u8(1))
        str_builder.append('1')

        return ''.join(str_builder)

    def get_moves(self):
        """
        Gets the legal moves of the current state

        :return: A List[Move] of legal moves
        """
        if self.moves:
            return self.moves
        return self._get_moves_new()

    def move(self, move:Move):
        """
        Executes a move and updates all the parameters of the Chessboard

        :param move: Type Move, represents the move to be executed
        :return: True if game is drawed after move is executed, False otherwise
        """
        
        if self.is_valid_move(move):
            # update piececount for _update_no_progress()
            self.piece_count = self._get_piece_count()
            self.pawns = self._get_pawns()

            self._update_enpassant()
            self._update_repetitions()
            self._update_bitboards(move)
            self._update_no_progress()
            self._update_player()
            self._update_move_counter()

            self.moves = []

            return True
        
        return False
    
    def is_valid_move(self, move:Move):
        """
        Checks if a move is valid

        :param move: Type Move, move to check legality of
        :return: True if move is valid, false otherwise
        """

        moves = self.get_moves()
        return move in moves

    def unmove(self):
        self._update_player()
        self._reverse_move_counter()
        self.bitboards = self.repetitions_list.pop()
        self.no_progress_counter.pop()
        self.enpassante = self.enpassant_list.pop()

    def get(self):
        """
        :return: shape (2,6) dtype=u64 ndarray where [i,j] represents the bitboard of Color i and Piece j
        """
        return self.bitboards

    def get_game_status(self):
        """
        Gets the status of the game in the current state

        :return: 0 if white wins, 1 if black wins, 2 if game is draw, 3 if game is ongoing
        """
    
        moves = self.get_moves()
        if not len(moves):
            return self.player_to_move
        if self._check_repetitions():
            return 2
        if self._check_no_progress():
            return 2
        return 3

    def is_draw(self):
        """
        Checks if the game is a draw or not based on repetitions and no-progress

        :return: True if game is draw, False otherwise
        """

        if self._check_repetitions():
            return True
        if self._check_no_progress():
            return True
        return False

    def get_fen(self):
        """
        :return: The FEN representation of the state of the chessboard
        """
        bit = u64(1)

        fen = ""

        board = ""
        bb = ls(bit, u8(63))
        for rank in range(8):
            empty_counter = 0
            for file in range(8):
                occupied = False
                for player in Color:
                    for piece_type in Piece:
                        if b_and(self.bitboards[player, piece_type], bb):
                            if empty_counter:
                                board += str(empty_counter)
                                empty_counter = 0
                            board += self._color_piece_type_to_char(player, piece_type)
                            occupied = True
                if not occupied:
                    empty_counter += 1
                if file==7 and empty_counter:
                    board += str(empty_counter)
                bb = rs(bb, bit)
            if rank!=7:
                board += '/'
        fen += board + " "

        player = self._player_to_char(self.player_to_move)
        fen += player + " "

        enpassante_index = u8(0)
        enpassante_bb = self.enpassante
        if enpassante_bb:
            while enpassante_bb:
                if b_and(enpassante_bb, bit):
                    break
                enpassante_index += 1
                enpassante_bb = rs(enpassante_bb, bit)
            enpassante = self._index_to_alg_sq(enpassante_index)
        else:
            enpassante = '-'
        fen += enpassante + " "

        no_progress = str(self.no_progress_counter[-1])
        fen += no_progress + " "
        
        move_counter = str(self.move_counter)
        fen += move_counter

        return fen

    def translate_board(self):
        # The outer structure is an 8 by 8 grid representing 'squares' of the board
        # the innermost list is meant to be represented as followed: (the 17 planes of the input repr.)
        # [If W pawn, If W knight, If W bishop, If W rook, If W queen, If W king, ...
        #  If B pawn, If B knight, If B bishop, If B rook, If B queen, If B king, ...
        #  Repetition Count for W as Integer, Repetition Count for B as Integer, ...
        #  1 If White, 0 If black (color), ...
        #  No Progress Count as Integer, ...
        #  1 if the grid square is an en passant square, 0 if not
        representation = [[
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []],
            [[], [], [], [], [], [], [], []]
        ]]
        # the full shape of the array is in the form of (1, 8, 8, 17)
        # technically not that efficient with these 4 nested for loops, however
        # the total looping is only around 750 loops (2*6*8*8) which is mostly fine
        for player in self.bitboards:
            for piece_type in player:
                for i in range(0, 8):
                    for l in range(0, 8):
                        position = b_and(lookup.rank_nr_list[i], lookup.rank_l_list[l])
                        if b_and(piece_type, position) != 0:
                            representation[0][i][l].append(1)
                        else:
                            representation[0][i][l].append(0)

        # repetitions
        repetitions_once = 1
        if any([all([all([self.bitboards[c][p]==r[c][p] for p in Piece]) for c in Color]) for r in self.repetitions_list]):
            repetitions_twice = 1
        else:
            repetitions_twice = 0

        if self.player_to_move == Color.WHITE:
            color = 0
        else:
            color = 1

        no_progress = self.no_progress_counter[-1]
        en_passant = self.enpassante
        for i in range(0, 8):
            for l in range(0, 8):
                representation[0][i][l].append(repetitions_once)
                representation[0][i][l].append(repetitions_twice)
                representation[0][i][l].append(color)
                representation[0][i][l].append(no_progress)
                position = b_and(lookup.rank_nr_list[i], lookup.rank_l_list[l])
                if b_and(en_passant, position) != 0:
                    representation[0][i][l].append(1)
                else:
                    representation[0][i][l].append(0)

        #for row in range(0,8):
            #print(representation[0][row])
        return array(representation)

    def _check_repetitions(self):
        if self._get_repetitions() == 2:
            return True
        return False
    
    def _check_no_progress(self):
        if self.no_progress_counter[-1] == 50:
            return True
        return False

    def _init_fen(self, fen:str):
        arr = fen.split()

        board = arr[0].split('/')
        numbers = "12345678"
        bit = u64(1)

        index = u64(63)
        for rank in board:
            for c in rank:
                if c in numbers:
                    index -= u64(c)
                else:
                    bb_indexes = self._char_to_bb_indexes(c)
                    self.bitboards[bb_indexes] = b_or(self.bitboards[bb_indexes], ls(bit, index))
                    index -= bit

        player = arr[1]
        self.player_to_move = self._char_to_player(player)
        if self.player_to_move == Color.WHITE:
            self.not_player_to_move = Color.BLACK
        else:
            self.not_player_to_move = Color.WHITE

        enpassante = arr[2]
        if enpassante == '-':
            self.enpassante = u64(0)
        else:
            self.enpassante = ls(bit, u8(alg_sq_to_index(enpassante)))

        no_progress = arr[3]
        self.no_progress_counter.append(u8(no_progress))

        move_counter = arr[4]
        self.move_counter = u8(move_counter)

    def _char_to_player(self, c:str):
        if c=='w':
            return Color.WHITE
        elif c=='b':
            return Color.BLACK
        else:
            raise RuntimeError("Invalid character as player color, %s" % c)
        
    def _char_to_bb_indexes(self, c:str):
        if c == 'P':
            return (Color.WHITE, Piece.PAWN)
        elif c == 'N':
            return (Color.WHITE, Piece.KNIGHT)
        elif c == 'B':
            return (Color.WHITE, Piece.BISHOP)
        elif c == 'R':
            return (Color.WHITE, Piece.ROOK)
        elif c == 'Q':
            return (Color.WHITE, Piece.QUEEN)
        elif c == 'K':
            return (Color.WHITE, Piece.KING)
        elif c == 'p':
            return (Color.BLACK, Piece.PAWN)
        elif c == 'n':
            return (Color.BLACK, Piece.KNIGHT)
        elif c == 'b':
            return (Color.BLACK, Piece.BISHOP)
        elif c == 'r':
            return (Color.BLACK, Piece.ROOK)
        elif c == 'q':
            return (Color.BLACK, Piece.QUEEN)
        elif c == 'k':
            return (Color.BLACK, Piece.KING)
        else:
            raise RuntimeError("Invalid character as Color and Piece, %s" % c)

    def _index_to_alg_sq(self, index:u8):
        rank = str(rs(index, u8(3))+1)
        file = b_and(index, u8(0b000_111))
        file = chr(65+(7-file))
        return file + rank

    def _player_to_char(self, color:Color):
        if color == Color.WHITE:
            return 'w'
        elif color == Color.BLACK:
            return 'b'
        else:
            raise RuntimeError("Invalid Color, %s" % str(color))

    def _color_piece_type_to_char(self, color:Color, piece_type:Piece):
        if color == Color.WHITE:
            if piece_type == Piece.PAWN:
                return 'P'
            elif piece_type == Piece.KNIGHT:
                return 'N'
            elif piece_type == Piece.BISHOP:
                return 'B'
            elif piece_type == Piece.ROOK:
                return 'R'
            elif piece_type == Piece.QUEEN:
                return 'Q'
            elif piece_type == Piece.KING:
                return 'K'
            else:
                raise RuntimeError("Invalid Piece, %s" % str(piece_type))
        elif color == Color.BLACK:
            if piece_type == Piece.PAWN:
                return 'p'
            elif piece_type == Piece.KNIGHT:
                return 'n'
            elif piece_type == Piece.BISHOP:
                return 'b'
            elif piece_type == Piece.ROOK:
                return 'r'
            elif piece_type == Piece.QUEEN:
                return 'q'
            elif piece_type == Piece.KING:
                return 'k'
            else:
                raise RuntimeError("Invalid Piece, %s" % str(piece_type))
        else:
            raise RuntimeError("Invalid Color, %s" % str(color))

    def _update_bitboards(self, move:Move):
        src_bb = ls(u64(1), move.src_index)
        dst_bb = ls(u64(1), move.dst_index)
        promotion_type = move.promotion_type

        enpassante = self.enpassante
        self.enpassante = u64(0)

        for piece_type in Piece:
            self.bitboards[self.not_player_to_move, piece_type] = b_and(self.bitboards[self.not_player_to_move, piece_type], b_not(dst_bb))
            if b_and(self.bitboards[self.player_to_move, piece_type], src_bb):
                if piece_type == Piece.PAWN:
                    self._move_pawn(src_bb, dst_bb, enpassante, promotion_type)
                else:
                    self._move_piece(src_bb, dst_bb, piece_type)

    def _update_enpassant(self):
        self.enpassant_list.append(self.enpassante)

    def _move_pawn(self, src_bb:u64, dst_bb:u64, enpassante:u64, promotion_type:Piece):
        self.bitboards[self.player_to_move, Piece.PAWN] = b_xor(self.bitboards[self.player_to_move, Piece.PAWN], src_bb)
        if promotion_type:
            self._move_promote(dst_bb, promotion_type)
        elif b_and(dst_bb, enpassante):
            self._move_enpassante(dst_bb)
        elif b_and(src_bb, rs(dst_bb, u64(16))) or b_and(src_bb, ls(dst_bb, u64(16))):
            self._move_pawn_double(dst_bb)
        else:
            self.bitboards[self.player_to_move, Piece.PAWN] = b_or(self.bitboards[self.player_to_move, Piece.PAWN], dst_bb)

    def _move_promote(self, dst_bb:u64, promotion_type:Piece):
        self.bitboards[self.player_to_move, promotion_type] = b_or(self.bitboards[self.player_to_move, promotion_type], dst_bb)

    def _move_pawn_double(self, dst_bb):
        self.bitboards[self.player_to_move, Piece.PAWN] = b_or(self.bitboards[self.player_to_move, Piece.PAWN], dst_bb)
        if self.player_to_move == Color.WHITE:
            self.enpassante = rs(dst_bb, u64(8))
        else:
            self.enpassante = ls(dst_bb, u64(8))

    def _move_enpassante(self, dst_bb:u64):
        self.bitboards[self.player_to_move, Piece.PAWN] = b_or(self.bitboards[self.player_to_move, Piece.PAWN], dst_bb)
        if self.player_to_move == Color.WHITE:
            self.bitboards[self.not_player_to_move, Piece.PAWN] = b_xor(self.bitboards[self.not_player_to_move, Piece.PAWN], rs(dst_bb, u64(8)))
        else:
            self.bitboards[self.not_player_to_move, Piece.PAWN] = b_xor(self.bitboards[self.not_player_to_move, Piece.PAWN], ls(dst_bb, u64(8)))

    def _move_piece(self, src_bb:u64, dst_bb:u64, piece_type:Piece):
        self.bitboards[self.player_to_move, piece_type] = b_xor(self.bitboards[self.player_to_move, piece_type], src_bb)
        self.bitboards[self.player_to_move, piece_type] = b_or(self.bitboards[self.player_to_move, piece_type], dst_bb)

    def _update_repetitions(self):
        self.repetitions_list.append(self.bitboards.copy())

    def _update_no_progress(self):
        if self._update_piece_count() or self._update_pawns():
            self.no_progress_counter.append(u8(0))
        else:
            self.no_progress_counter.append(self.no_progress_counter[-1]+u8(1))

    def _update_piece_count(self):
        new_piece_count = self._get_piece_count()
        if new_piece_count != self.piece_count:
            self.piece_count = new_piece_count
            return True
        return False
    
    def _update_pawns(self):
        new_pawns = self._get_pawns()
        if new_pawns != self.pawns:
            self.pawns = new_pawns
            return True
        return False

    def _get_pawns(self):
        return b_or(self.bitboards[Color.WHITE, Piece.PAWN], self.bitboards[Color.BLACK, Piece.PAWN])
    
    def _update_player(self):
        tmp = self.player_to_move
        self.player_to_move = self.not_player_to_move
        self.not_player_to_move = tmp

    def _update_move_counter(self):
        if self.player_to_move == Color.WHITE:
            self.move_counter += u8(1)

    def _reverse_move_counter(self):
        if self.player_to_move == Color.BLACK:
            self.move_counter -= u8(1)

    def _get_piece_count(self):
        bit = u8(1)
        count = 0
        for player in self.bitboards:
            for piece_bb in player:
                while piece_bb:
                    if b_and(piece_bb, bit):
                        count += 1
                    piece_bb = rs(piece_bb, bit)
        return count
    
    def _get_repetitions(self):
        repetitions = 0
        for state in self.repetitions_list:
            if self._bbs_equal(state, self.bitboards):
                repetitions += 1
        return repetitions

    def _bbs_equal(self, bbs1:ndarray, bbs2:ndarray):
        for player in Color:
            for piece_type in Piece:
                if bbs1[player, piece_type] != bbs2[player, piece_type]:
                    return False
        return True

    def _combine_bb(self):
        # combines bb of both player colors seperately and together
        index = 0
        for player in self.bitboards:
            combined = u64(0)
            for bb in player:
                combined = b_or(combined, bb)
            self.combined[index] = combined
            index += 1

        self.combined[2] = b_or(self.combined[Color.WHITE], self.combined[Color.BLACK])

    def _get_moves_new(self):
        """
        Generates the legal moves of the current state

        :return: A List[Move] of legal moves
        """

        # combines the piece_type bitboards of both colors seperately, used in later calculations
        self._combine_bb()
        # vars to hold legal moves and takes
        moves:List[Move] = []
        takes:List[Move] = []
        # get takes by different piece types
        moves += self._get_moves()

        for m in moves:
            if m.is_take:
                takes.append(m)

        if len(takes):
            return takes

        return moves

    def _get_moves_by_bb(self, src_index:u8, dst_bb:u64, takes:bool=False):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = u8(0)
        bit = u8(1)
        while dst_bb:
            if b_and(dst_bb, bit):
                moves.append(Move(src_index, dst_index, is_take=takes))
            dst_index += bit
            dst_bb = rs(dst_bb, bit)
        return moves
    
    def _get_moves_by_bb_pawn(self, src_index:u8, dst_bb:u64, takes:bool=False):
        # converts moves represented by a source index and a destination bb to
        # moves represented by the Move() class and returns them.
        moves:List[Move] = []
        dst_index = u8(0)
        bit = u8(1)
        while dst_bb:
            if b_and(dst_bb, bit):
                if (dst_index < 8) or (56 <= dst_index):
                    for piece_type in (Piece):
                        if piece_type != Piece.PAWN:
                            moves.append(Move(src_index, dst_index, piece_type, takes))
                else:
                    moves.append(Move(src_index, dst_index, is_take=takes))
            dst_index += bit
            dst_bb = rs(dst_bb, bit)
        return moves
    
    def _get_moves(self):
        # get moves by the self.player_to_move kings bb and returns them as [Move]
        moves:List[Move] = []
        bit = u8(1)
        for piece_type in Piece:
            src_index = uint(0)
            src_bb = self.bitboards[self.player_to_move, piece_type]
            while src_bb:
                if b_and(src_bb, bit):
                    moves += self._get_moves_by_piece_type(piece_type, src_index)
                src_index += bit
                src_bb = rs(src_bb, bit)
        return moves
    
    def _get_moves_by_piece_type(self, piece_type:Piece, src_index:u8):
        if piece_type == Piece.PAWN:
            if self.player_to_move == Color.WHITE:
                return self._get_moves_pawn_white(src_index)
            elif self.player_to_move == Color.BLACK:
                return self._get_moves_pawn_black(src_index)
            else:
                raise RuntimeError("Invalid player_to_move: %s" % str(self.player_to_move))
        elif piece_type == Piece.KNIGHT:
            return self._get_moves_knight(src_index)
        elif piece_type == Piece.BISHOP:
            return self._get_moves_bishop(src_index)
        elif piece_type == Piece.ROOK:
            return self._get_moves_rook(src_index)
        elif piece_type == Piece.QUEEN:
            return self._get_moves_queen(src_index)
        elif piece_type == Piece.KING:
            return self._get_moves_king(src_index)
        else:
            raise RuntimeError("Invalid piece_type: %s" % str(piece_type))

    def _get_moves_file(self, src_index:u8):
        moves:List[Move] = []
        shift = b_and(src_index, u8(0b000_111))
        rank_index = rs(b_and(src_index, u8(0b111_000)), u8(3))

        occ = rs(self.combined[2], shift)
        occ = b_and(occ, lookup.FILE_H_BB)
        occ *= lookup.DIAG_MASKS[7]
        occ = rs(occ, u8(0b111_000))

        dst_bb = lookup.FILE_H_ATTACKS[rank_index, occ]
        dst_bb = ls(dst_bb, shift)

        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_rank(self, src_index:u8):
        moves:List[Move] = []
        rank_index = b_and(src_index, u8(0b000_111))
        shift:u8 = b_and(src_index, u8(0b111_000))

        occ = u8(rs(self.combined[2], shift))

        dst_bb = u64(lookup.FIRST_RANK_ATTACKS[rank_index, occ])
        dst_bb = ls(dst_bb, shift)
        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_diag(self, src_index:u8):
        moves:List[Move] = []
        rank_index = b_and(src_index, u8(0b000_111))

        occ = b_and(self.combined[2], lookup.DIAG_MASKS[src_index])
        occ *= lookup.FILE_H_BB
        occ = rs(occ, u8(0b111_000))

        dst_bb = lookup.FIRST_RANK_ATTACKS[rank_index, occ]
        dst_bb *= lookup.FILE_H_BB
        dst_bb = b_and(dst_bb, lookup.DIAG_MASKS[src_index])

        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_antidiag(self, src_index:u8):
        moves:List[Move] = []
        rank_index = b_and(src_index, u8(0b000_111))

        occ = b_and(self.combined[2], lookup.ANTIDIAG_MASKS[src_index])
        occ *= lookup.FILE_H_BB
        occ = rs(occ, u8(0b111_000))

        dst_bb = lookup.FIRST_RANK_ATTACKS[rank_index, occ]
        dst_bb *= lookup.FILE_H_BB
        dst_bb = b_and(dst_bb, lookup.ANTIDIAG_MASKS[src_index])

        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))

        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_pawn_white(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        src_bb = ls(u64(1), src_index)

        takes_bb = b_and(ls(src_bb, u8(9)), b_not(lookup.FILE_H_BB))
        takes_bb = b_or(takes_bb, b_and(ls(src_bb, u8(7)), b_not(lookup.FILE_A_BB)))
        takes_bb = b_and(takes_bb, b_or(self.combined[Color.BLACK], self.enpassante))

        if takes_bb: # if takes avaiable, no need to find non-takes
            return self._get_moves_by_bb_pawn(src_index, takes_bb, takes=True)

        moves_bb = ls(src_bb, u8(8))
        if (8 <= src_index < 16):
            moves_bb = b_or(moves_bb, ls(src_bb, u8(16)))
        moves_bb = b_and(moves_bb, b_not(b_or(self.combined[2], ls(b_and(b_not(src_bb), self.combined[2]), u8(8)))))

        bb = b_or(moves_bb, takes_bb)

        return self._get_moves_by_bb_pawn(src_index, bb)

    def _get_moves_pawn_black(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        src_bb = ls(u64(1), src_index)

        takes_bb = b_and(rs(src_bb, u8(7)), b_not(lookup.FILE_H_BB))
        takes_bb = b_or(takes_bb, b_and(rs(src_bb, u8(9)), b_not(lookup.FILE_A_BB)))
        takes_bb = b_and(takes_bb, b_or(self.combined[Color.WHITE], self.enpassante))

        if takes_bb: # if takes avaiable, no need to find non-takes
            return self._get_moves_by_bb_pawn(src_index, takes_bb, takes=True)

        moves_bb = rs(src_bb, u8(8))
        if (48 <= src_index < 56):
            moves_bb = b_or(moves_bb, rs(src_bb, u8(16)))
        moves_bb = b_and(moves_bb, b_not(b_or(self.combined[2], rs(b_and(b_not(src_bb), self.combined[2]), u8(8)))))

        bb = b_or(moves_bb, takes_bb)

        return self._get_moves_by_bb_pawn(src_index, bb)
    
    def _get_moves_knight(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        moves:List[Move] = []
        dst_bb = lookup.KNIGHT_BB[src_index]
        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))
        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves

    def _get_moves_bishop(self, src_index:u8):
        moves:List[Move] = []
        moves += self._get_moves_diag(src_index)
        moves += self._get_moves_antidiag(src_index)
        return moves

    def _get_moves_rook(self, src_index:u8):
        moves:List[Move] = []
        moves += self._get_moves_rank(src_index)
        moves += self._get_moves_file(src_index)
        return moves
    
    def _get_moves_queen(self, src_index:u8):
        moves:List[Move] = []
        moves += self._get_moves_diag(src_index)
        moves += self._get_moves_antidiag(src_index)
        moves += self._get_moves_rank(src_index)
        moves += self._get_moves_file(src_index)
        return moves

    def _get_moves_king(self, src_index:u8):
        # given a src_index, generate moves and returns them as [Move]
        moves:List[Move] = []
        dst_bb = lookup.KING_BB[src_index]
        takes_bb = b_and(dst_bb, self.combined[self.not_player_to_move])
        moves_bb = b_and(dst_bb, b_not(self.combined[2]))
        moves += self._get_moves_by_bb(src_index, takes_bb, takes=True)
        moves += self._get_moves_by_bb(src_index, moves_bb)
        return moves