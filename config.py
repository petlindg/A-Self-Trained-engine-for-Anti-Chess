'''
===========================================
Global configuration values for the program
===========================================
'''
from chess import Color, Piece

'constant defining exploration vs exploitation of the tree'
exploration_constant: float = 4
'constant defining how many iterations per tree will be performed'
tree_iterations: int = 800

'output dimensions for the neural network'
output_representation = (1,1,8,8,76)

'maximum number of games saved in the FIFO queue that the model uses as a dataset'
max_buffer_size = 100

'total number of iterations that are performed before the training stops'
training_iterations = 500

'total number of games that are performed per training iteration'
games_per_iteration = 5

'epochs during the training process'
epochs = 30

'batch size for the training process'
batch_size = 16

'Learning rate for the network'
learning_rate = 0.2
'verbosity of the training process (how much terminal output is displayed)'
verbosity = 0

checkpoint_path = "checkpoints/checkpoint.ckpt"

'evaluation method is the method by which the MCTS calculates P values for nodes during the expansion phase'
'if the model is not designated as an argument. Can be one of: ones, dirichlet'
evaluation_method = 'dirichlet'

piece_list = [(Color.BLACK, Piece.KING), (Color.WHITE, Piece.ROOK)]