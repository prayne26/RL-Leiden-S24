import keras
from keras import layers
from tensorflow.keras.optimizers import Adam

class DeepNeuralNetwork():
    def __init__(self, input_size, output_size, layers, neurons_per_layer, learning_rate, exploration):
        self.lr = learning_rate
    #     #super(FullyConnectedModel, self).__init__()
          
        
    def custom_network(self, input_size, num_actions):
        model = keras.Sequential([
                    layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=input_size),
                    layers.Conv2D(64, 4, strides=2, activation="relu"),
                    layers.Conv2D(64, 3, strides=1, activation="relu"),
                    layers.Flatten(),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(num_actions, activation="linear"),
                ])
        
        model.compile(loss='mse', optimizer=Adam(lr=self.lr), metrics=['accuracy', 'mse'])
        return model