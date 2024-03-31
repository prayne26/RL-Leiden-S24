import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import sys

class DeepNeuralNetwork():
    def __init__(self, input_size, output_size, learning_rate, num_layers=1, neurons_per_layer=[32]):
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
            return
        
        model = keras.Sequential()
        model.add(layers.Dense(16, activation='relu', kernel_initializer='he_uniform', input_dim=self.input_size, name='L0'))
        
        for l in range(self.n_layers):
            for npl in self.neurons_per_layer:
                model.add(layers.Dense(npl, activation='relu', kernel_initializer='he_uniform', name='L'+str(l+1)))
                   
                   
        model.add(layers.Dense(self.output_size, activation='linear', kernel_initializer='he_uniform'))
              
        model.compile(loss='mse', optimizer=Adam(lr=self.lr), metrics=['accuracy', 'mse'])
        return model