import keras
from keras import layers,optimizers,losses


class DeepNeuralNetwork:
    def __init__(self, nr_layers=1, neurons_per_layer=16):
        # super(DeepNeuralNetwork, self).__init__()
        '''Dense Neural Network
        params
        input_size = shape of input array
        layers = number of
        '''
        self.nr_layers = nr_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_size = (1, 4)
        self.num_actions = 2

    def custom_Qnetwork(self, verbose=False):
        network = keras.Sequential()
        network.add(keras.layers.Flatten(input_shape=self.input_size))
        for i in range(self.nr_layers):
            network.add(keras.layers.Dense(self.neurons_per_layer, activation='relu'))
        network.add(keras.layers.Dense(self.num_actions, activation='linear'))
        if verbose:
            print(network.summary())
        network.compile(optimizer=keras.optimizers.adam_v2,loss=keras.losses.binary_crossentropy)
        return network
