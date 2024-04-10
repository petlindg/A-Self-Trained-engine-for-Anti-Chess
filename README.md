### A Self Trained Engine for Antichess

This project is a bachelor thesis project for students at the ... section? 
at Chalmers University of Technology. 

## Configuration instructions
## Installation instructions
## Operating instructions

## Files

This is the file structure of our project:

Folder: assets
    b_bishop.png
    b_king.png
    b_knight.png
    b_pawn.png
    b_queen.png
    b_rook.png
    w_bishop.png
    w_king.png
    w_knight.png
    w_pawn.png
    w_queen.png
    w_rook.png

Folder: Game
    dataset_generator.py
    Player.py
    state_generator.py
    TrainingGame.py
    Utils.py

.gitignore
chess.perft.py
chess.py
chess.test.py
config.py
evalForcedGame.py
genDatasetForcedGame.py
interface.py
LICENSE.md
logger.py
main.py
nn_architecture.py
node.py
printDataset.py
README.md (this file)
testing.py
training.py

# Folder: assets 

This folder contains images provided by wikipedia user Cburnett from this website:

 https://commons.wikimedia.org/wiki/Category:PNG_chess_pieces/Standard_transparent

# Folder: Game

   # dataset_generator.py
   # Player.py

   This is the player class. It has methods which are used to calculate the best move from the state it is currently in, and methods to run and update the Monte Carlo tree.
   
   # state_generator.py
   # TrainingGame.py
   # Utils.py

# chess.perft.py
# chess.py



# chess.test.py

This is the test file for the file chess.py, which includes the classes Chessboard and Move. It containts three testing classes: TestMoveGen, TestGameState and TestTranslations. The class TestMoveGen is testing for a number of different types of moves and captures if the correct number of generated possible moves is correct, and also if invalid moves are generated or valid moves are omitted. The TestGameState class tests whether the Chessboard class recognizes different special ending cases of the game, namely stalemates, no progress draws, and move repetition draws. The TestTranslations class tests whether the FEN representation translations work correctly in the Chessboard class.

# config.py

From this file, the user may control global variable values of the program. This includes parameters of the tree search algorithm and parameters when training the neural network. These may be tweaked to suit you preferences, but the values we used in our implementation are left as a default.

# evalForcedGame.py
# genDatasetForcedGame.py
# interface.py
# LICENSE.md

This is the license file of our project.

# logger.py
# main.py
# nn_architecture.py
# node.py
# printDataset.py
# README.md (this file)
# testing.py
# training.py

## Copyright and licensing information

This project is licensed under the terms of the GNU-AGPL license.

## Contact information for the distributor or author

Abdulrazak Ahmed Mohamed
Email: ahmedmoh@chalmers.se

Alexander Petersson
Email: alepete@chalmers.se

Daniel Skarehag
Email: gusskarda@student.gu.se

Liam Kral
Email: liamk@chalmers.se

Josef Jakobson
Email: josefjak@chalmers.se

Petter Lindgren
Email: petlindg@chalmers.se


A list of known bugs[3]
Troubleshooting instructions[3]

## Credits and acknowledgments

We would like to give cretit and thanks to our tutor Jean-Philippe Bernardy for giving us guidance and help 
throughout the project. 

## A changelog (usually aimed at fellow programmers)

Börja vid MVP och nämn updateringar efter det? 

## A news section (usually aimed at end users)

samma fråga som för changelog
