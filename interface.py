# piece assets provided by wikepedia user Cburnett from
# https://commons.wikimedia.org/wiki/Category:PNG_chess_pieces/Standard_transparent



# -----------------------------
# ----------- TO DO -----------
# -----------------------------
# make pieces dragable
# implement translation function from logical board to ChessboardGUI
# implement translation function from logical move to (x1, y1, x2, y2) as a tkinter grid
# add special case castling
# add special case en-passante
# add special case promotion
# fix when selecting same colored piece that it doesn't take and instead reselects

from tkinter import *
from typing import List
import numpy as np
import sys
sys.path.insert(0, 'D:/gits/A-Self-Trained-engine-for-Anti-Chess')      # need to define path locally until we merge modules properly
from chess import Chessboard as cb

# macro definitions
WINDOW_SIZE    = 560     
ASSET_PATH     = "assets/"
WHITE          = "#FFFFFF"     # white
WHITE_SELECTED = "#C0C0C0"     # darker white
BLACK          = "#808080"     # gray
BLACK_SELECTED = "#505050"     # darker gray

class SquareGUI(Canvas):
    # visual properties
    size:int            # size of square in pixels
    bg:str              # background color of square
    bg_selected:str      # background color of square when square is selected
    piece_img:PhotoImage # rendered image of piece on the square
    piece_img_id:int      # id of rendered image
    # logical properties
    pos_x:int            # position of square
    pos_y:int
    selected:bool       # square selected to be moved
    piece_color:str      # color of the piece on the square
    piece_type:str       # type of the piece on the square

    def __init__(self, parent, size:int, pos_x:int, pos_y:int):
        # visual construction
        self.bg=(BLACK if ((pos_x+pos_y)%2) else WHITE)
        super().__init__(parent, width=size, height=size, bg=self.bg, highlightthickness=0)
        self.size=size
        self.bg_selected=(BLACK_SELECTED if ((pos_x+pos_y)%2) else WHITE_SELECTED)
        # logical construction
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.selected=False
        self.piece_color="none"
        self.piece_type="none"
        # set position of square in root window
        self.grid(row=pos_x, column=pos_y)
        # bind square to left-click
        self.bind("<Button-1>", parent.on_m1)
    # constructor for piece
    def set_piece(self, piece_color:str, piece_type:str):
        self.piece_color=piece_color
        self.piece_type=piece_type
        self.piece_img=self._getAsset()
        if(self.piece_img):
            self.piece_img_id = self.create_image(self.size/2, self.size/2, image=self.piece_img, anchor=CENTER)
    # deconstructor for piece
    def delete_piece(self):
            self.piece_color="none"
            self.piece_type="none"
            self.delete(self.piece_img_id)
    def select(self):
        self.selected=True
        self.config(bg=self.bg_selected)
    def deselect(self):
        self.selected=False
        self.config(bg=self.bg)
    # gets the rendered file as a PhotoImage object
    def _getAsset(self):
        if self.piece_color == "white":
            if self.piece_type == "pawn":
                return PhotoImage(file=ASSET_PATH+"w_pawn.png")
            if self.piece_type == "knight":
                return PhotoImage(file=ASSET_PATH+"w_knight.png")
            if self.piece_type == "bishop":
                return PhotoImage(file=ASSET_PATH+"w_bishop.png")
            if self.piece_type == "rook":
                return PhotoImage(file=ASSET_PATH+"w_rook.png")
            if self.piece_type == "queen":
                return PhotoImage(file=ASSET_PATH+"w_queen.png")
            if self.piece_type == "king":
                return PhotoImage(file=ASSET_PATH+"w_king.png")
        elif self.piece_color == "black":
            if self.piece_type == "pawn":
                return PhotoImage(file=ASSET_PATH+"b_pawn.png")
            if self.piece_type == "knight":
                return PhotoImage(file=ASSET_PATH+"b_knight.png")
            if self.piece_type == "bishop":
                return PhotoImage(file=ASSET_PATH+"b_bishop.png")
            if self.piece_type == "rook":
                return PhotoImage(file=ASSET_PATH+"b_rook.png")
            if self.piece_type == "queen":
                return PhotoImage(file=ASSET_PATH+"b_queen.png")
            if self.piece_type == "king":
                return PhotoImage(file=ASSET_PATH+"b_king.png")


class ChessboardGUI(Tk):
    piece_selected : bool            # true if any square is selected
    selected_piece : SquareGUI       # alias to the selected piece
    board : List[List[SquareGUI]]   # main grid used for the board
    board_size : int                 # size of one side of the square board
    square_size : int                # size of a square

    def __init__(self, size):
        super().__init__()                          # init root window
        self.title("Chessboard")                    # sets window title
        self.geometry(str(size) + "x" + str(size))  # sets window to square of size {size}
        self.resizable(False,False)                 # makes window unresizable in x and y
        self.board_size=size
        self.square_size=size/8
        self.piece_selected = False
        # initializes the squares in self.board
        self.board = [[SquareGUI(self, size=self.square_size, pos_x=col, pos_y=row) for row in range(8)] for col in range(8)]
        # bind root window to right-click
        self.bind("<Button-3>", self.on_m2)
    def on_m1(self, event):
        if self.piece_selected:
            if event.widget.selected == False:
                if(self.selected_piece.piece_type!="none"):
                    self.move(self.selected_piece.pos_x, self.selected_piece.pos_y, event.widget.pos_x, event.widget.pos_y)
                self.deselect_all()
            else:
                event.widget.deselect()
            self.piece_selected = False
        else:
            if(event.widget.piece_type!="none"):
                event.widget.select()
                self.selected_piece = event.widget
                self.piece_selected=True
    def on_m2(self, event):
        self.deselect_all()
    def move(self, posx1:int, posy1:int, posx2:int, posy2:int, promoteTo=None):
        self.board[posx2][posy2].set_piece(self.board[posx1][posy1].piece_color, self.board[posx1][posy1].piece_type)
        self.board[posx1][posy1].delete_piece()
    def set_selected(self, piece:SquareGUI):
        self.piece_selected = True
        self.selected_piece = piece
    def deselect_all(self):
        self.piece_selected = False
        [[self.board[i][j].deselect() for i in range(8)] for j in range(8)]
    def init_board(self, bitboards):
        # init white pieces
        self._initBoard_byType(bitboards[0, 0], "white", "pawn")
        self._initBoard_byType(bitboards[0, 1], "white", "knight")
        self._initBoard_byType(bitboards[0, 2], "white", "bishop")
        self._initBoard_byType(bitboards[0, 3], "white", "rook")
        self._initBoard_byType(bitboards[0, 4], "white", "queen")
        self._initBoard_byType(bitboards[0, 5], "white", "king")
        # init black pieces
        self._initBoard_byType(bitboards[1, 0], "black", "pawn")
        self._initBoard_byType(bitboards[1, 1], "black", "knight")
        self._initBoard_byType(bitboards[1, 2], "black", "bishop")
        self._initBoard_byType(bitboards[1, 3], "black", "rook")
        self._initBoard_byType(bitboards[1, 4], "black", "queen")
        self._initBoard_byType(bitboards[1, 5], "black", "king")
    def _initBoard_byType(self, bitboard:np.uint64, color:str, piece_type:str):
        i = 7
        j = 7
        while bitboard:
            if bitboard & np.uint64(1):
                self.board[i][j].set_piece(color, piece_type)
            j-=1
            if j<0:
                j=7
                i-=1
            bitboard >>= np.uint8(1)
    # initializes default chess starting state, temporary until ruleset and translation from ruleset is complete
    def fill_board(self):
        #init back-rank black
        self.board[0][0].set_piece("black", "rook")
        self.board[0][1].set_piece("black", "knight")
        self.board[0][2].set_piece("black", "bishop")
        self.board[0][3].set_piece("black", "queen")
        self.board[0][4].set_piece("black", "king")
        self.board[0][5].set_piece("black", "bishop")
        self.board[0][6].set_piece("black", "knight")
        self.board[0][7].set_piece("black", "rook")
        #init black pawns
        [self.board[1][i].set_piece("black", "pawn") for i in range(8)]
        #init white pawns
        [self.board[6][i].set_piece("white", "pawn") for i in range(8)]
        #init back-rank white
        self.board[7][0].set_piece("white", "rook")
        self.board[7][1].set_piece("white", "knight")
        self.board[7][2].set_piece("white", "bishop")
        self.board[7][3].set_piece("white", "queen")
        self.board[7][4].set_piece("white", "king")
        self.board[7][5].set_piece("white", "bishop")
        self.board[7][6].set_piece("white", "knight")
        self.board[7][7].set_piece("white", "rook")


a = cb()
a.init_board_standard()

board = ChessboardGUI(WINDOW_SIZE)
board.init_board(a.bitboards)

board.mainloop()