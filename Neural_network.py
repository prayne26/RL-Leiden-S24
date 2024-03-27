import keras
from keras import layers

class DeepNeuralNetwork():
    def __init__(self, input_size, output_size, layers, neurons_per_layer, learning_rate, exploration):
        #super(FullyConnectedModel, self).__init__()
        pass

    def custom_Qnetwork(self):
        num_actions = 2
        return keras.Sequential(
            [layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=(4, 84, 84)),
            layers.Conv2D(64, 4, strides=2, activation="relu"),
            layers.Conv2D(64, 3, strides=1, activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_actions, activation="linear"),
            ]
        )