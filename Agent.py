import gymnasium as gym
import numpy as np
import random
import os
from collections import deque
from Helper import argmax, softmax
from Neural_network import DeepNeuralNetwork
import tensorflow as tf


class DQNAgent:
  def __init__(self, learning_rate, gamma, policy, batch_size, RB_bool=True, TNN_bool=True, temp=None, epsilon=None,
               nlp=None):
    # Parameters
    if nlp is None:
      nlp = [32, 32]
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.policy = policy  # one of the Boltzmann of Egreddy
    self.temp = temp
    self.epsilon = epsilon
    self.training_counts = 0
    self.batch_size = batch_size
    self.weights_updating_frequency = 30
    self.num_episodes = 200

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
      with open(path + file_name, "a") as myfile:
        myfile.write(log)
    except:
      print("Unable to save to files.")

  def remember(self, state, action, reward, next_state, done):
    self.replay_buffer.append((state, action, reward, next_state, done))

  def act(self, state):
    if self.policy == 'egreedy':
      if self.epsilon is None:
        raise KeyError("Provide an epsilon")

      p = np.random.random()
      if p < self.epsilon:
        a = np.random.randint(0, self.n_actions)
      else:
        a = argmax(self.nn_target.predict(state, verbose=0))

    elif self.policy == 'softmax':
      if self.temp is None:
        raise KeyError("Provide a temperature")

      a = np.random.choice(range(self.n_actions), 1, p=softmax(self.nn_Q.predict(state), self.temp))[0]

    return a

  def sample_from_replay_memory(self):
    return random.sample(self.replay_buffer, min(self.batch_size,len(self.replay_buffer)))

  def train(self):
    self.training_counts += 1
    batch_sample = self.sample_from_replay_memory()
    for state, action, reward, next_state, done in batch_sample:
      if done:
        target = reward
      else:
        target = reward + self.gamma * np.amax(self.nn_target.predict(next_state, verbose=0))
      current_prediction = self.nn_Q.predict(state, verbose=0)
      #print(f'pred:{current_prediction}')
      #print(f'action: {action}, target: {target}')
      current_prediction[0][action] = target
      #print(f'newpred:{current_prediction}')
      #np.expand_dims(target, axis=(0, 1))
      result = self.nn_Q.fit(state, current_prediction, epochs=1, verbose=0)

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
        i += 1

        if self.RB_bool:
          self.remember(state, action, reward, next_state, done)

        self.train()
        #loss.append(result.history['loss'])

        if self.TNN_bool and self.training_counts % self.weights_updating_frequency == 0 and i > 0:
          print(f'updating target network: {self.training_counts}')
          self.nn_target.set_weights(self.nn_Q.get_weights())

        state = next_state
        steps.append(i)

      scores.append(score)
      #loss_avg.append(np.mean(loss))

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
