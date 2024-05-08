# piece assets provided by wikepedia user Cburnett from
# https://commons.wikimedia.org/wiki/Category:PNG_chess_pieces/Standard_transparent



# -----------------------------
# ----------- TO DO -----------
# -----------------------------
# make pieces dragable
# implement translation function from logical move to (x1, y1, x2, y2) as a tkinter grid
# fix a better way to communicate with the chessboard, for example winchecking and some popup on win

from tkinter import *
from typing import List
import numpy as np
from chess import Chessboard as cb
from chess import Move

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
        if self.piece_type != "none":
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

class PromotionWindowGUI(Toplevel):
    def __init__(self, parent, color, x, y):
        super().__init__()
        self.geometry(f"{int(WINDOW_SIZE*5/8)}x{int(WINDOW_SIZE/8)}+{x}+{y}")
        promotion_squares = [SquareGUI(self, WINDOW_SIZE/8, 0, pos_y) for pos_y in range(5)]
        promotion_squares[0].set_piece(color, "knight")
        promotion_squares[1].set_piece(color, "bishop")
        promotion_squares[2].set_piece(color, "rook")
        promotion_squares[3].set_piece(color, "queen")
        promotion_squares[4].set_piece(color, "king")
        self.overrideredirect(True)
        self.parent = parent

    def on_m1(self, event):
        self.parent.on_m1(event)

class ChessboardGUI(Tk):
    piece_selected : bool            # true if any square is selected
    selected_piece : SquareGUI       # alias to the selected piece
    board : List[List[SquareGUI]]   # main grid used for the board
    board_size : int                 # size of one side of the square board
    square_size : int                # size of a square

    dst_piece : SquareGUI
    awaiting_promo : bool
    promotion_window : PromotionWindowGUI

    def __init__(self, size, send_move, get_bitboards):
        super().__init__()                          # init root window
        self.title("Chessboard")                    # sets window title
        self.geometry(str(size) + "x" + str(size))  # sets window to square of size {size}
        self.resizable(False,False)                 # makes window unresizable in x and y
        self.board_size=size
        self.square_size=size/8
        self.piece_selected = False
        self.awaiting_promo = False
        # initializes the squares in self.board
        self.board = [[SquareGUI(self, size=self.square_size, pos_x=col, pos_y=row) for row in range(8)] for col in range(8)]
        # function to send moves to the logical board
        self.send_move = send_move
        # function to get the logical board
        self.get_bitboards = get_bitboards
        # bind root window to right-click
        self.bind("<Button-3>", self.on_m2)
    def on_m1(self, event):
        if self.awaiting_promo:
            self.try_promo(self.selected_piece, self.dst_piece, event.widget.piece_type)
            self._close_choose_promo()
            return

        dst:SquareGUI = event.widget
        if self.piece_selected:
            src = self.selected_piece
            if dst.selected == False:
                if (src.piece_type == "pawn"):
                    if (src.piece_color == "white") and (dst.pos_x==0):
                        self.dst_piece = dst
                        self._open_choose_promo(event, self.selected_piece.piece_color)
                    elif (src.piece_color == "black") and (dst.pos_x==7):
                        self.dst_piece = dst
                        self._open_choose_promo(event, self.selected_piece.piece_color)
                    else:
                        self.try_move(src, dst)
                else:
                    self.try_move(src, dst)  
                self.deselect_all()
            else:
                dst.deselect()
            self.piece_selected = False
        else:
            if(dst.piece_type!="none"):
                dst.select()
                self.selected_piece = event.widget
                self.piece_selected=True
    def on_m2(self, event):
        self.deselect_all()
    def try_move(self, src_sq:SquareGUI, dst_sq:SquareGUI):
        src_index = 7-src_sq.pos_y+8*(7-src_sq.pos_x)
        dst_index = 7-dst_sq.pos_y+8*(7-dst_sq.pos_x)

        if self.send_move(Move(np.uint64(src_index), np.uint64(dst_index))):
            print("updating bitboards")
            self.init_board(self.get_bitboards())
            self.update()
            self.init_board(self.get_bitboards())
    def try_promo(self, src_sq:SquareGUI, dst_sq:SquareGUI, promotion_type:str):
        src_index = 7-src_sq.pos_y+8*(7-src_sq.pos_x)
        dst_index = 7-dst_sq.pos_y+8*(7-dst_sq.pos_x)

        p = 0
        if promotion_type == "knight":
            p = 1
        elif promotion_type == "bishop":
            p = 2
        elif promotion_type == "rook":
            p = 3
        elif promotion_type == "queen":
            p = 4
        elif promotion_type == "king":
            p = 5
        
        if self.send_move(Move(src_index, dst_index, p)):
            self.init_board(self.get_bitboards())

    def move(self, posx1:int, posy1:int, posx2:int, posy2:int, promoteTo=None):
        self.board[posx2][posy2].set_piece(self.board[posx1][posy1].piece_color, self.board[posx1][posy1].piece_type)
        self.board[posx1][posy1].delete_piece()
    def set_selected(self, piece:SquareGUI):
        self.piece_selected = True
        self.selected_piece = piece
    def deselect_all(self):
        self.piece_selected = False
        [[self.board[i][j].deselect() for i in range(8)] for j in range(8)]
    def clear_board(self):
        for i in range(8):
            for j in range(8):
                if self.board[i][j].piece_type != "none":
                    self.board[i][j].delete_piece()
    def init_board(self, bitboards):
        self.clear_board()
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
    def _open_choose_promo(self, event, color):
        self.grab_set()
        self.promotion_window = PromotionWindowGUI(self, color, event.x_root, event.y_root)
        self.awaiting_promo = True
    def _close_choose_promo(self):
        self.grab_release()
        self.awaiting_promo = False
        self.promotion_window.destroy()


class Game():
    def __init__(self):
        self.state = cb()
        self.GUI   = ChessboardGUI(WINDOW_SIZE, self.state.move, self.state.get)

def main():
    game = Game()

    game.GUI.init_board(game.state.bitboards)

    game.GUI.mainloop()

if __name__ == "__main__":
    main()