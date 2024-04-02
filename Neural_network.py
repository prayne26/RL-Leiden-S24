import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
import sys

class DeepNeuralNetwork():
    def __init__(self, input_size, output_size, learning_rate, num_layers, neurons_per_layer):
        self.lr = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = num_layers
        self.neurons_per_layer = neurons_per_layer  # npl       

        
