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

        
    def custom_network(self):
        if len(self.neurons_per_layer) != self.n_layers:
            print("Wrong number! More/less elements in the list neurons_per_layer then the layers number.")
            sys.exit() 
            return
        
        model = keras.Sequential()       
        for l,npl in zip(range(self.n_layers), self.neurons_per_layer):
            if  l == 0:
                model.add(layers.Dense(npl, activation='relu', kernel_initializer='he_uniform', input_dim=self.input_size, name="L"+str(l)))
            else:
                model.add(layers.Dense(npl, activation='relu', kernel_initializer='he_uniform', name="L"+str(l)))
                        
        model.add(layers.Dense(self.output_size, activation='linear', kernel_initializer='he_uniform'))
              
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['accuracy', 'mse'])
        return model