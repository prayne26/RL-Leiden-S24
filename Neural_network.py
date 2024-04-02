import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import sys

class DeepNeuralNetwork():
    def __init__(self, input_size, output_size, learning_rate, num_layers, neurons_per_layer):
        self.lr = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = num_layers
        self.neurons_per_layer = neurons_per_layer  # npl       

    # def GPU_check(self):
    #     print(K.tensorflow_backend._get_available_gpus())
    #     if K.tensorflow_backend._get_available_gpus():
            # print("Running on GPU.")
        
