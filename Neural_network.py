import keras
from keras import layers
from keras.optimizers import Adam
from keras import backend as K
import sys

class DeepNeuralNetwork():
    def __init__(self, input_size, output_size, learning_rate, num_layers, neurons_per_layer):
        self.lr = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = num_layers
        self.neurons_per_layer = neurons_per_layer          

    # def GPU_check(self):
    #     print(K.tensorflow_backend._get_available_gpus())
    #     if K.tensorflow_backend._get_available_gpus():
            # print("Running on GPU.")
        
    def custom_network(self):
        if len(self.neurons_per_layer) != self.n_layers:
            print("Wrong number! More/less elements in the list neurons_per_layer then the layers number.")
            sys.exit()
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_size,)))
        for l in range(self.n_layers):
            model.add(layers.Dense(self.neurons_per_layer[l], activation='relu', kernel_initializer='he_uniform'))
                   
                   
        model.add(layers.Dense(self.output_size, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['accuracy', 'mse'])
        model.summary()
        return model