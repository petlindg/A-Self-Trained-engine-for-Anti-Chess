from tkinter import *
from typing import List

# -----------------------------
# ----------- TO DO -----------
# -----------------------------
# make pieces movable
# add comments
# fix piece images, fix error handling
# add some test

assetPath:str="interface/assets/"

WHITE = "#FFFFFF"
WHITE_SELECTED = "#C0C0C0"
BLACK = "#808080"
BLACK_SELECTED = "#505050"

class SquareGUI(Canvas):
    posx:int
    posy:int
    dragged:bool
    selected:bool
    bg:str
    bgSelected:str
    size:int
    pieceImg:PhotoImage
    pieceImgId:int
    pieceColor:str
    pieceType:str

    def __init__(self, parent, size:int, posx:int, posy:int):
        self.bg=(BLACK if ((posx+posy)%2) else WHITE)
        super().__init__(parent, width=size, height=size, bg=self.bg, highlightthickness=0)
        self.size=size
        self.bgSelected=(BLACK_SELECTED if ((posx+posy)%2) else WHITE_SELECTED)
        self.posx=posx
        self.posy=posy
        self.dragged=False
        self.selected=False
        self.pieceColor="none"
        self.pieceType="none"
        self.grid(row=posx, column=posy)

        self.bind("<Button-1>", self.toggleSelect)

    def _getAsset(self):
        match self.pieceColor:
            case "white":
                match self.pieceType:
                    case "pawn":
                        return PhotoImage(file=assetPath+"w_pawn.png")
                    case "knight":
                        return PhotoImage(file=assetPath+"w_knight.png")
                    case "bishop":
                        return PhotoImage(file=assetPath+"w_bishop.png")
                    case "rook":
                        return PhotoImage(file=assetPath+"w_rook.png")
                    case "queen":
                        return PhotoImage(file=assetPath+"w_queen.png")
                    case "king":
                        return PhotoImage(file=assetPath+"w_king.png")
            case "black":
                match self.pieceType:
                    case "pawn":
                        return PhotoImage(file=assetPath+"b_pawn.png")
                    case "knight":
                        return PhotoImage(file=assetPath+"b_knight.png")
                    case "bishop":
                        return PhotoImage(file=assetPath+"b_bishop.png")
                    case "rook":
                        return PhotoImage(file=assetPath+"b_rook.png")
                    case "queen":
                        return PhotoImage(file=assetPath+"b_queen.png")
                    case "king":
                        return PhotoImage(file=assetPath+"b_king.png")
    def setPiece(self, pieceColor:str, pieceType:str):
        self.pieceColor=pieceColor
        self.pieceType=pieceType
        self.pieceImg=self._getAsset()
        if(self.pieceImg):
            self.pieceImgId = self.create_image(self.size/2, self.size/2, image=self.pieceImg, anchor=CENTER)
    def deletePiece(self):
            self.pieceColor="none"
            self.pieceType="none"
            self.delete(self.pieceImgId)
    def toggleSelect(self, event):
        self.selected=not self.selected
        if(self.selected):
            self.config(bg=self.bgSelected)
        else:
            self.config(bg=self.bg)
    def deselect(self):
        self.selected=False
        self.config(bg=self.bg)


class ChessboardGUI(Tk):
    board : List[List[SquareGUI]]
    boardSize : int
    squareSize : int
    def __init__(self, size):
        super().__init__()
        self.boardSize=size
        self.squareSize=size/8
        self.title("Chessboard")
        self.geometry(str(size) + "x" + str(size))
        self.resizable(False,False)
        self.board = [[SquareGUI(self, size=self.squareSize, posx=col, posy=row) for row in range(8)] for col in range(8)]

        self.bind("<Button-3>", self.deselectAll)

    def deselectAll(self, event):
        [[self.board[i][j].deselect() for i in range(8)] for j in range(8)]
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
    def move(self, posx1:int, posy1:int, posx2:int, posy2:int, promoteTo=None):
        self.board[posx2][posy2].setPiece(self.board[posx1][posy1].pieceColor, self.board[posx1][posy1].pieceType)
        self.board[posx1][posy1].deletePiece()

def hello(event):
    print("Hello")
    return

windowSize=560
board = ChessboardGUI(windowSize)
board.fillBoard()

board.mainloop()