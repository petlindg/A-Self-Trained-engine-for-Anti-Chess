import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Input, Conv2D, LeakyReLU, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import config

# NEURAL NETWORK INPUTS
# 12 planes for the pieces (2 players of 6 pieces)
# 2 planes for repetition
# 1 plane for color
# 1 plane for no-progress (not neccessary i think)
# 1 plane for the en passant square

n = 8  # board size

# non boolean values: pieces for every player + the no-progress
# boolean values: color, is repitition, the square for en passant
total_input_planes = (2 * 6 + 1) + (1 + 2 + 1)
INPUT_SHAPE = (n, n, total_input_planes)

# NEURAL NETWORK OUTOUTS

# the model has 2 outputs: policy and value
# ouput_shape[0] should be the number of possible moves
# 8x8 board: 8*8=64 possible actions
# Queen-like moves for each of the 64 squares, moving in 8 directions up to 7 squares = 56
# Knight moves for each of the 64 squares, with 8 possible moves each
# Underpromotions considering pawn can promote on any of the 64 squares,in 3 directions (forward, capture left, capture right),
# to 5 possible pieces (knight, bishop, rook, queen, king)
#   total values: 8*8*(56+8+15) = 5056
# ouput_shape[1] should be 1: a scalar value (v)

# 79 planes for chess:
total_planes = 56 + 8 + 15
# the output shape for the vector
OUTPUT_SHAPE = (8 * 8 * total_planes, 1)

# NEURAL NETWORK PARAMETERS

LEARNING_RATE = config.learning_rate
# filters for the convolutional layers
CONVOLUTION_FILTERS = 256
# amount of hidden residual layers according to the alpha zero paper
RESIDUAL_BLOCKS = config.RESIDUAL_BLOCKS

KERNEL_INITIALIZER = 'glorot_normal'


class NeuralNetwork:
    def __init__(self, input_shape: tuple, output_shape: tuple):
        """
        Initializes a Neural Network for predicting move probabilities and winning chances in a game from the current state.

        Parameters:
        - input_shape (tuple): Shape of the input (board state).
        - output_shape (tuple): Shape of the output (move probabilities and win probability).
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_hidden_layers = RESIDUAL_BLOCKS
        self.convolution_filters = CONVOLUTION_FILTERS

    def convolutional_layer(self, input_x, name) -> KerasTensor:
        conv_layer = Conv2D(self.convolution_filters,
                            kernel_size=(3, 3),
                            name=f'{name}',
                            padding="same",
                            kernel_initializer=KERNEL_INITIALIZER
                            )(input_x)

        conv_layer = BatchNormalization(axis=1, name=f'{name}_BatchNorm')(conv_layer)
        conv_layer = Activation('relu', name=f'{name}_relu')(conv_layer)

        return conv_layer

    def residual_layer(self, input_x, name) -> KerasTensor:
        conv1 = self.convolutional_layer(input_x, f'res_block_{name}_conv_1')
        conv2 = Conv2D(self.convolution_filters,
                       kernel_size=(3, 3),
                       name=f'residual_block_{name}_conv_2',
                       padding="same",
                       kernel_initializer=KERNEL_INITIALIZER
                       )(conv1)

        conv2 = BatchNormalization(axis=1, name=f'res_block_{name}_BatchNorm')(conv2)
        # skip connections
        """
        used to add the input of the block to its output.
        This allows the creation of a "shortcut" or "skip" connection that
        bypasses the layers between the input and output of the residual block

        """
        skip_connection = Add(name=f'res_block_{name}_skip')([conv2, input_x])
        res_layer = Activation('relu', name=f'res_block_{name}_relu')(skip_connection)
        return res_layer

    def policy_network(self, x) -> Model:
        x = Conv2D(2, (1, 1),
                   kernel_initializer=KERNEL_INITIALIZER,
                   padding='same',
                   name='policy_1_conv_input')(x)

        x = BatchNormalization(axis=1, name='policy_2_BatchNorm')(x)

        x = Activation('relu', name='policy_3_relu')(x)

        x = Flatten(name='policy_4_flatten')(x)

        x = Dense(self.output_shape[0], activation='softmax', name='policy_output')(x)
        return x

    def value_network(self, x) -> Model:
        x = Conv2D(1,
                   (1, 1),
                   padding='same',
                   name='value_1_conv_input',
                   input_shape=(self.convolution_filters, *self.input_shape[1:]),
                   kernel_initializer=KERNEL_INITIALIZER)(x)

        x = BatchNormalization(axis=1, name='value_2_BatchNorm')(x)

        x = Activation('relu', name='value_3_relu')(x)

        x = Flatten(name='value_4_flatten')(x)

        x = Dense(self.convolution_filters,
                  name='value_5_linear',
                  activation='relu',
                  kernel_initializer=KERNEL_INITIALIZER)(x)

        x = Dense(1, activation='sigmoid',
                  name='value_output',
                  kernel_initializer=KERNEL_INITIALIZER)(x)
        return x

    def build_nn(self) -> Model:
        """
        this is for building the neural network architecture and integrating the policy and value heads
        returns Keras Model object with the specified inputs, outputs, and training configuration
        """

        input_feature = Input(shape=self.input_shape, name='input_feature')

        # Initial convolutional layer
        x = self.convolutional_layer(input_feature, 'input')

        # Add residual layers
        for i in range(self.num_hidden_layers):
            x = self.residual_layer(x, str(i + 1))

        # Instantiate the model with inputs and both outputs
        model = Model(inputs=input_feature, outputs=[self.policy_network(x), self.value_network(x)])

        # Compile the model with specified losses, optimizer, and loss weights
        model.compile(
            loss=[
                'categorical_crossentropy',
                # measures the disparity between the actual distribution of the labels and the predicted probabilities.
                'mean_squared_error'
                # uses the gradients of the loss function to update the model's weights in a way that minimizes the loss
            ],
            # optimizer=Adam(learning_rate=LEARNING_RATE),
            # loss_weights={
            #    'policy_head': 0.5,  # Equal weighting for policy and value losses
            #    'value_head': 0.5
            # }
        )

        return model


if __name__ == "__main__":
    # create the model
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    print(model.summary())





