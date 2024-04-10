'''
===========================================
Global configuration values for the program
===========================================
'''

'constant defining exploration vs exploitation of the tree'
exploration_constant: float = 4
'constant defining how many iterations per tree will be performed'
tree_iterations: int = 200

'output dimensions for the neural network'
output_representation = (1,1,8,8,76)

'maximum number of games saved in the FIFO queue that the model uses as a dataset'
max_buffer_size = 100

'total number of iterations that are performed before the training stops'
training_iterations = 500

'total number of games that are performed per training iteration'
games_per_iteration = 50

'epochs during the training process'
epochs = 30

'batch size for the training process'
batch_size = 16

'verbosity of the training process (how much terminal output is displayed)'
verbosity = 0

checkpoint_path = "checkpoints/checkpoint.ckpt"

'if the training will generate random states based on piece list'
random_state_generation = False

'proportion of the training data that will be in the training data set'
train_split = 0.8

'evaluation is a boolean for if the program will run with a single thread and print out all debugging output'
evaluation = False

'number of processes for the multiprocessing'
processes = 100

'batch size of the multiprocessing neural network, dictates how many states the network will predict at a time'
nn_batch = 150
if evaluation:
    nn_batch = 1