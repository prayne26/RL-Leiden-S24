import gymnasium as gym
import numpy as np
import random
import os
from collections import deque
from Helper import argmax, softmax
from Neural_network import DeepNeuralNetwork
import tensorflow as tf

class DQNAgent:
  def __init__(self, learning_rate, gamma, policy, batch_size, RB_bool=True, TNN_bool=True, temp=None, epsilon=None, nlp=[128,128,128]):
    # Parameters
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.policy = policy # one of the Boltzmann of Egreddy
    self.temp = temp
    self.epsilon = epsilon
    self.training_counts = 0
    self.batch_size = batch_size
    self.weights_updating_frequancy = 10
    self.num_episodes = 1000

    self.RB_bool = RB_bool
    self.TNN_bool = TNN_bool


    # Enviorment
    self.env = gym.make('CartPole-v1')
    self.n_actions = self.env.action_space.n
    self.n_states = self.env.observation_space.shape[0]


    # Q and target nural network
    self.nn = DeepNeuralNetwork(self.n_states, self.n_actions, self.learning_rate, len(nlp), nlp)
    self.nn_Q = self.nn.custom_network()
    self.nn_target = self.nn.custom_network()
    self.nn_target.set_weights(self.nn_Q.get_weights())
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    # Replay buffer stroing (state, action, reward, next_state, done) as doubly ended queue
    self.replay_buffer = deque(maxlen=10000)

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

  def remeber(self, state, action, reward, next_state, done):
    self.replay_buffer.append((state, action, reward, next_state, done))
  
  def act(self, state):
    if self.policy == 'egreedy':
      if self.epsilon is None:
        raise KeyError("Provide an epsilon")
            
      p = np.random.random()
      if p < self.epsilon:
        a = np.random.randint(0, self.n_actions)
      else:
        a = argmax(self.nn_Q.predict(state))
                        
    elif self.policy == 'softmax':
      if self.temp is None:
          raise KeyError("Provide a temperature")
   
      a = np.random.choice(range(self.n_actions), 1, p=softmax(self.nn_Q.predict(state), self.temp))[0]
        
    return a
  
  def sample_from_replay_memory(self):
    return random.sample(self.replay_buffer, self.batch_size)
  
  def train(self):
    self.training_counts += 1
    if len(self.replay_buffer) < self.batch_size:
            return

    batch_sample = self.sample_from_replay_memory()
    states = np.zeros((self.batch_size, self.n_states))
    next_states = np.zeros((self.batch_size, self.n_states))
    actions, rewards, dones = [], [], []
    
    j=0
    for x in batch_sample:
      states[j] = x[0]
      next_states[j] = x[3]
      j+=1

      actions.append(x[1])
      rewards.append(x[2])
      dones.append(x[4])

    target = self.nn_Q.predict(states)
    target_next = self.nn_target.predict(next_states)

    for i in range(self.batch_size):
      if dones[i]:
        target[i][actions[i]] = rewards[i]
      else:
        target[i][actions[i]] = rewards[i] + self.gamma * (np.amax(target_next[i]))

    res = self.nn_Q.fit(states, target, batch_size=self.batch_size)
    return res


  def run(self):
    print("Starting running...")
    loss_avg, scores, steps = [], [], []
    for e in range(self.num_episodes):
      loss = []
      score = 0
      state, _ = self.env.reset()
      state = np.array(state).reshape(1, self.n_states)
      done, i = False, 0

      while not done:
        action = self.act(state)
        next_state, reward, done, info, _ = self.env.step(action)
        next_state = np.array(next_state).reshape(1, self.n_states)
        score += reward
 
        i+=1

        if self.RB_bool:
          self.remeber(state, action, reward, next_state, done)

        if len(self.replay_buffer) >= self.batch_size and self.training_counts%self.weights_updating_frequancy == 0:
          result = self.train()
          loss.append(result.history['loss'])

        if self.TNN_bool and done:
          self.nn_target.set_weights(self.nn_Q.get_weights())

        state = next_state
        steps.append(i)

      scores.append(score) 
      loss_avg.append(np.mean(loss))

      log = "Episode: {}/{}, Total reward: {}, Total steps: {}, Parameters: epsilon={}, lr={}.\n".format(e, 
                                                                                     self.num_episodes, 
                                                                                     score, 
                                                                                     steps[-1],
                                                                                     self.epsilon,
                                                                                     self.learning_rate)
      self.save_log(log)
      print(log)
    
    print('Scores: ', scores)
    print('Steps: ', steps)
    return loss_avg, steps


