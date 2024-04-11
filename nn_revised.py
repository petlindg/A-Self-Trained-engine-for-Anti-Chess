import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Activation, BatchNormalization, Input, Conv2D, Dense, Flatten, Add, GlobalAveragePooling2D, Multiply, Reshape, DepthwiseConv2D
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import config

# Assuming config.py contains definitions for learning_rate and RESIDUAL_BLOCKS
LEARNING_RATE = config.learning_rate
RESIDUAL_BLOCKS = 13
CONVOLUTION_FILTERS = 256
  # Adjusted for the number of possible moves in chess

KERNEL_INITIALIZER = 'glorot_normal'
# 79 planes for chess:
total_planes = 56 + 8 + 15
# the output shape for the vector
OUTPUT_SHAPE = (8 * 8 * total_planes, 1)


n = 8  # board size

# non boolean values: pieces for every player + the no-progress
# boolean values: color, is repitition, the square for en passant
total_input_planes = (2 * 6 + 1) + (1 + 2 + 1)
INPUT_SHAPE = (n, n, total_input_planes)



class NeuralNetwork:
    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_hidden_layers = RESIDUAL_BLOCKS
        self.convolution_filters = CONVOLUTION_FILTERS

    def SqueezeAndExcitation(self, inputs, ratio=2): 
        channel_axis = -1  # Typically, channels are last in TensorFlow
        c = inputs.shape[channel_axis]  # Get the number of channels from the input tensor
        x = GlobalAveragePooling2D()(inputs) 
        x = Dense(c // ratio, activation="relu", use_bias=False)(x) 
        x = Dense(c, activation="sigmoid", use_bias=False)(x) 
        x = Reshape((1, 1, c))(x)  # Reshape to the correct shape
        out = Multiply()([inputs, x])  # Use broadcasting for element-wise multiplication
        return out

    def convolutional_layer(self, input_x, name) -> KerasTensor:
        conv_layer = Conv2D(self.convolution_filters, kernel_size=(3, 3), name=f'{name}', padding="same", kernel_initializer=KERNEL_INITIALIZER)(input_x)
        conv_layer = BatchNormalization(axis=1, name=f'{name}_BatchNorm')(conv_layer)
        conv_layer = Activation('relu', name=f'{name}_relu')(conv_layer)
        return conv_layer

    def residual_se_layer(self, input_x, name, x, use_se = False) -> KerasTensor:
        conv0 = Conv2D(128+64*x, kernel_size=(1, 1), name=f'res_se_block_{name}_conv_2', padding="same", kernel_initializer=KERNEL_INITIALIZER)(input_x)
        conv0 = BatchNormalization(axis=1, name=f'{name}_BatchNorm')(conv0)
        conv0 = Activation('relu', name=f'{name}_relu')(conv0)

        conv1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(conv0)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(self.convolution_filters, kernel_size=(1, 1), name=f'{name}', padding="same", kernel_initializer=KERNEL_INITIALIZER)(conv1)
        conv2 = BatchNormalization(axis=1)(conv2)
        

        if use_se:
            se = self.SqueezeAndExcitation(conv2)
        else:
            se = conv2
        
        skip_connection = Add(name=f'res_se_block_{name}_skip')([se, input_x])
        res_se_layer = Activation('relu', name=f'res_se_block_{name}_relu')(skip_connection)
        return res_se_layer

    def policy_head(self, x) -> KerasTensor:
        x = Conv2D(self.convolution_filters, (3, 3), padding='same',kernel_initializer=KERNEL_INITIALIZER, name='policy_head_conv1')(x)
        x = BatchNormalization(axis=1, name='policy_head_bn1')(x)
        x = Activation('relu', name='policy_head_relu1')(x)
        
        # Second convolution to transform dimensions to 79x8x8
        x = Conv2D(79, (3, 3), padding='same', name='policy_head_conv2', kernel_initializer=KERNEL_INITIALIZER)(x)
        
        # Flatten the tensor to a vector and cut down to the required size (assuming 1858 legal moves)
        x = Flatten(name='policy_head_flatten')(x)
        policy_output = Activation('softmax', name='policy_output')(x)
        return policy_output
    
    def value_head(self, x) -> KerasTensor:
        x = Conv2D(8, (1, 1),padding='same',
                   name='value_1_conv_input',
                   kernel_initializer=KERNEL_INITIALIZER)(x)
        x = BatchNormalization(axis=1, name='value_head_bn1')(x)
        x = Activation('relu', name='value_head_relu1')(x)
        
        
        x = Flatten(name='value_head_flatten')(x)
        x = Dense(256, activation='relu', name='value_head_dense1')(x)

        value_output = Dense(1, activation='sigmoid', name='value_output_classical')(x)

        return value_output
    
 


    def build_nn(self) -> Model:
        input_feature = Input(shape=self.input_shape, name='input_feature')
        
        

        # Initial convolutional layer
        x = self.convolutional_layer(input_feature, 'input')

        # Add residual layers
        for i in range(self.num_hidden_layers):
            x = self.residual_se_layer(x, str(i + 1), i, use_se=(i > 8))

          

        model = Model(inputs=input_feature, outputs=[self.policy_head(x), self.value_head(x)])
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer,
                     loss=[
                'categorical_crossentropy',
                # measures the disparity between the actual distribution of the labels and the predicted probabilities.
                'mean_squared_error'
                # uses the gradients of the loss function to update the model's weights in a way that minimizes the loss
            ])

        return model


if __name__ == "__main__":
    # create the model
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    print(model.summary())

   