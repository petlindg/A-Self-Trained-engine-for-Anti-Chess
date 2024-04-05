from config import checkpoint_path
from nn_architecture import NeuralNetwork, INPUT_SHAPE, OUTPUT_SHAPE


def main():
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)
    model.save('saved_model/model')




if __name__ == "__main__":
    main()