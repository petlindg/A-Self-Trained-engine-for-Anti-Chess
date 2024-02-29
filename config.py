'''
===========================================
Global configuration values for the program
===========================================
'''

'constant defining exploration vs exploitation of the tree'
exploration_constant: float = 4
'constant defining how many iterations per tree will be performed'
tree_iterations: int = 200

output_representation = (1,1,8,8,76)

max_buffer_size = 100

training_iterations = 5

games_per_iteration = 5

epochs = 30

batch_size = 16

verbosity = 0

checkpoint_path = "checkpoints/checkpoint.ckpt"