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

from tkinter import *
from typing import List

# macro definitions
WINDOW_SIZE     = 560     
ASSET_PATH:str  = "interface/assets/"
WHITE           = "#FFFFFF"     # white
WHITE_SELECTED  = "#C0C0C0"     # darker white
BLACK           = "#808080"     # gray
BLACK_SELECTED  = "#505050"     # darker gray

class SquareGUI(Canvas):
    # visual properties
    size:int            # size of square in pixels
    bg:str              # background color of square
    bgSelected:str      # background color of square when square is selected
    pieceImg:PhotoImage # rendered image of piece on the square
    pieceImgId:int      # id of rendered image
    # logical properties
    posx:int            # position of square
    posy:int
    selected:bool       # square selected to be moved
    pieceColor:str      # color of the piece on the square
    pieceType:str       # type of the piece on the square

    def __init__(self, parent, size:int, posx:int, posy:int):
        # visual construction
        self.bg=(BLACK if ((posx+posy)%2) else WHITE)
        super().__init__(parent, width=size, height=size, bg=self.bg, highlightthickness=0)
        self.size=size
        self.bgSelected=(BLACK_SELECTED if ((posx+posy)%2) else WHITE_SELECTED)
        # logical construction
        self.posx=posx
        self.posy=posy
        self.selected=False
        self.pieceColor="none"
        self.pieceType="none"
        # set position of square in root window
        self.grid(row=posx, column=posy)
        # bind square to left-click
        self.bind("<Button-1>", parent.on_m1)
    # constructor for piece
    def setPiece(self, pieceColor:str, pieceType:str):
        self.pieceColor=pieceColor
        self.pieceType=pieceType
        self.pieceImg=self._getAsset()
        if(self.pieceImg):
            self.pieceImgId = self.create_image(self.size/2, self.size/2, image=self.pieceImg, anchor=CENTER)
    # deconstructor for piece
    def deletePiece(self):
            self.pieceColor="none"
            self.pieceType="none"
            self.delete(self.pieceImgId)
    def select(self):
        self.selected=True
        self.config(bg=self.bgSelected)
    def deselect(self):
        self.selected=False
        self.config(bg=self.bg)
    # gets the rendered file as a PhotoImage object
    def _getAsset(self):
        match self.pieceColor:
            case "white":
                match self.pieceType:
                    case "pawn":
                        return PhotoImage(file=ASSET_PATH+"w_pawn.png")
                    case "knight":
                        return PhotoImage(file=ASSET_PATH+"w_knight.png")
                    case "bishop":
                        return PhotoImage(file=ASSET_PATH+"w_bishop.png")
                    case "rook":
                        return PhotoImage(file=ASSET_PATH+"w_rook.png")
                    case "queen":
                        return PhotoImage(file=ASSET_PATH+"w_queen.png")
                    case "king":
                        return PhotoImage(file=ASSET_PATH+"w_king.png")
            case "black":
                match self.pieceType:
                    case "pawn":
                        return PhotoImage(file=ASSET_PATH+"b_pawn.png")
                    case "knight":
                        return PhotoImage(file=ASSET_PATH+"b_knight.png")
                    case "bishop":
                        return PhotoImage(file=ASSET_PATH+"b_bishop.png")
                    case "rook":
                        return PhotoImage(file=ASSET_PATH+"b_rook.png")
                    case "queen":
                        return PhotoImage(file=ASSET_PATH+"b_queen.png")
                    case "king":
                        return PhotoImage(file=ASSET_PATH+"b_king.png")


class ChessboardGUI(Tk):
    pieceSelected : bool            # true if any square is selected
    selectedPiece : SquareGUI       # alias to the selected piece
    board : List[List[SquareGUI]]   # main grid used for the board
    boardSize : int                 # size of one side of the square board
    squareSize : int                # size of a square

    def __init__(self, size):
        super().__init__()                          # init root window
        self.title("Chessboard")                    # sets window title
        self.geometry(str(size) + "x" + str(size))  # sets window to square of size {size}
        self.resizable(False,False)                 # makes window unresizable in x and y
        self.boardSize=size
        self.squareSize=size/8
        self.pieceSelected = False
        # initializes the squares in self.board
        self.board = [[SquareGUI(self, size=self.squareSize, posx=col, posy=row) for row in range(8)] for col in range(8)]
        # bind root window to right-click
        self.bind("<Button-3>", self.on_m2)
    def on_m1(self, event):
        if self.pieceSelected:
            if event.widget.selected == False:
                if(self.selectedPiece.pieceType!="none"):
                    self.move(self.selectedPiece.posx, self.selectedPiece.posy, event.widget.posx, event.widget.posy)
                self.deselectAll()
            else:
                event.widget.deselect()
            self.pieceSelected = False
        else:
            if(event.widget.pieceType!="none"):
                event.widget.select()
                self.selectedPiece = event.widget
                self.pieceSelected=True
    def on_m2(self, event):
        self.deselectAll()
    def move(self, posx1:int, posy1:int, posx2:int, posy2:int, promoteTo=None):
        self.board[posx2][posy2].setPiece(self.board[posx1][posy1].pieceColor, self.board[posx1][posy1].pieceType)
        self.board[posx1][posy1].deletePiece()
    def setSelected(self, piece:SquareGUI):
        self.pieceSelected = True
        self.selectedPiece = piece
    def deselectAll(self):
        self.pieceSelected = False
        [[self.board[i][j].deselect() for i in range(8)] for j in range(8)]
    # initializes default chess starting state, temporary until ruleset and translation from ruleset is complete
    def fillBoard(self):
        #init back-rank black
        self.board[0][0].setPiece("black", "rook")
        self.board[0][1].setPiece("black", "knight")
        self.board[0][2].setPiece("black", "bishop")
        self.board[0][3].setPiece("black", "queen")
        self.board[0][4].setPiece("black", "king")
        self.board[0][5].setPiece("black", "bishop")
        self.board[0][6].setPiece("black", "knight")
        self.board[0][7].setPiece("black", "rook")
        #init black pawns
        [self.board[1][i].setPiece("black", "pawn") for i in range(8)]
        #init white pawns
        [self.board[6][i].setPiece("white", "pawn") for i in range(8)]
        #init back-rank white
        self.board[7][0].setPiece("white", "rook")
        self.board[7][1].setPiece("white", "knight")
        self.board[7][2].setPiece("white", "bishop")
        self.board[7][3].setPiece("white", "queen")
        self.board[7][4].setPiece("white", "king")
        self.board[7][5].setPiece("white", "bishop")
        self.board[7][6].setPiece("white", "knight")
        self.board[7][7].setPiece("white", "rook")


board = ChessboardGUI(WINDOW_SIZE)
board.fillBoard()

board.mainloop()