### A Self Trained Engine for Antichess

This project is a bachelor thesis project for students at Chalmers University of Technology and the University of Gothenburg. 

## Configuration instructions

You can configure the different parameters of the Monte Carlo Tree Search, and of the training of the neural network in the config.py file.

## Installation instructions

Run this command in the command prompt to download all necessary packages to use the antichess engine:
```
pip install -r requirements.txt
```
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

Folder: chess
    chess.perft.py
    chess.test.py
    chessboard.py
    lookup.py
    move.py
    utils.py
    
Folder: Game
    dataset_generator.py
    Player.py
    state_generator.py
    TrainingGame.py
    Utils.py

.gitignore
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

## Credits and acknowledgments

We would like to give cretit and thanks to our tutor Jean-Philippe Bernardy for giving us guidance and help 
throughout the project. 

