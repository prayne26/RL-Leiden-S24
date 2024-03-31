import gymnasium as gym
import numpy as np
import os
import random
from collections import deque
from Helper import argmax, softmax
from Neural_network import DeepNeuralNetwork
from tensorflow.keras import models, layers, optimizers

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, policy, learning_rate, gamma, epsilon, npl):
        '''npl - nurons per layer,, it will be a list with the numbers of nurons in the layers []'''
        self.n_state = state_size
        self.n_actions = action_size
        self.replay_buffer = deque(maxlen=2000)

        self.policy = policy
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        nn = DeepNeuralNetwork(state_size, action_size, learning_rate, len(npl), npl)
        self.model_Q = nn.custom_network() # main neural network
        self.model_T = nn.custom_network()# target neural network
        self.update_target_model()

        self.max_episodes = 1000
        self.max_steps = 500 # the envirment limit
        self.weights_updating_frequancy = 10

    def update_target_model(self):
        # Copy weights from the main model to target_model
        self.model_T.set_weights(self.model_Q.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
      if self.policy == 'egreedy':
        if self.epsilon is None:
          raise KeyError("Provide an epsilon")
              
        p = np.random.random()
        if p <= self.epsilon:
          a = np.random.randint(0, self.n_actions)
        else:
          a = argmax(self.model_Q.predict(state))
                          
      elif self.policy == 'softmax':
        if self.temp is None:
            raise KeyError("Provide a temperature")
    
        a = np.random.choice(range(self.n_actions), 1, p=softmax(self.model_Q.predict(state), self.temp))[0]
        
      return a
    
    def sample_from_replay_memory(self):
      return random.sample(self.replay_buffer, self.batch_size)

    def replay(self):
        minibatch = self.sample_from_replay_memory()
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + np.multiply(self.gamma, np.amax(self.model_T.predict(next_state)[0])))
            target_f = self.model_Q.predict(state)
            target_f[0][action] = target
            self.model_Q.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model_Q.load_weights(name)

    def save(self, name):
        self.model_Q.save_weights(name)

    def save_log(self, log):
      path = "Logs/"
      file_name = "log.txt"
      if not os.path.exists(path):
        os.makedirs(path)
      try:
        with open(path+file_name, "a") as myfile:
          myfile.write(log)
      except:
          print("Unable to save to files.")

    def run(self):
        print("Starting running...")
        env = gym.make('CartPole-v1')
        # scores = deque(maxlen=100)
        loss_avg, scores, steps = [], [], []
        for e in range(self.max_episodes):  # we may try diffrent criterion for stopping
            loss = []
            score = 0
            state, _ = env.reset()
            state = np.reshape(state, [1, self.n_state])
            for step in range(self.max_steps):  # CartPole-v1 enforced max step
                action = self.act(state)
                next_state, reward, done, info, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.n_state])

                score += reward

                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    log = "Episode: {}/{}, Total reward: {}, Total steps: {}, Parameters: epsilon={}, lr={}.\n".format(e, 
                                                                                     self.max_episodes, 
                                                                                     score, 
                                                                                     step,
                                                                                     self.epsilon,
                                                                                     self.learning_rate)
                    self.save_log(log)
                    print(log)
                    break
                
            # scores.append(step)
            # if len(scores) == 100 and np.mean(scores) >= 195.0:
            #     print(f"Solved after {e} episodes!")
            #     break
            scores.append(score) 
            loss_avg.append(np.mean(loss))

            if len(self.replay_buffer) > self.batch_size:
                self.replay()
            
            if e % self.weights_updating_frequancy == 0:
              self.update_target_model()

        env.close()

        print('Scores: ', scores)
        print('Steps: ', steps)
        return loss_avg, steps 

