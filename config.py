'''
===========================================
Global configuration values for the program
===========================================
'''
from chess.utils import Color, Piece
from math import sqrt
'constant defining exploration vs exploitation of the tree'
exploration_constant: float = sqrt(2)
'constant defining how many iterations per tree will be performed'
tree_iterations: int = 800

'output dimensions for the neural network'
output_representation = (1,1,8,8,79)

'maximum number of games saved in the FIFO queue that the model uses as a dataset'
max_buffer_size = 30000

'total number of iterations that are performed before the training stops'
training_iterations = 500

'total number of games that are performed per training iteration'
games_per_iteration = 20

'epochs during the training process'
epochs = 30

'batch size for the training process'
batch_size = 64

'Learning rate for the network'
learning_rate = 0.2
'verbosity of the training process (how much terminal output is displayed)'
verbosity = 1

checkpoint_path = "checkpoints/checkpoint.ckpt"

'evaluation method is the method by which the MCTS calculates P values for nodes during the expansion phase'
'if the model is not designated as an argument. Can be one of: ones, dirichlet'
evaluation_method = 'dirichlet'

piece_list = [(Color.WHITE, Piece.KING), (Color.WHITE, Piece.KING), (Color.WHITE, Piece.KING),
    (Color.WHITE, Piece.ROOK),
    (Color.BLACK, Piece.KING), (Color.BLACK, Piece.KING), (Color.BLACK, Piece.KING),
    (Color.BLACK, Piece.ROOK)]


RESIDUAL_BLOCKS = 13

'location to save training data'
TRAINING_DATA_PATH = "trainingData/"

checkpoint_path = "checkpoints/checkpoint.ckpt"

'if the training will generate random states based on piece list'
random_state_generation = False

'proportion of the training data that will be in the training data set'
train_split = 0.8

'evaluation is a boolean for if the program will run with a single thread and print out all debugging output'
evaluation = False

'number of processes for the multiprocessing'
processes = 200

'batch size of the multiprocessing neural network, dictates how many states the network will predict at a time'
nn_batch = 150
if evaluation:
    nn_batch = 1
