import gymnasium as gym
import numpy as np
import random
import os
from collections import deque
from Helper import argmax, softmax
from Neural_network import DeepNeuralNetwork

class DQNAgent:
  def __init__(self, learning_rate, gamma, policy, train_max, RB_bool=True, TNN_bool=True, temp=None, epsilon=None, nlp=[128,128,128]):
    # Parameters
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.policy = policy # one of the Boltzmann of Egreddy
    self.RB_bool = RB_bool
    self.TNN_bool = TNN_bool
    self.temp = temp
    self.epsilon = epsilon
    self.training_counts = 0
    self.train_max = train_max
    self.weights_updating_frequancy = 100


    # Enviorment
    self.env = gym.make('CartPole-v1')
    self.n_actions = self.env.action_space.n
    self.n_states = self.env.observation_space.shape[0]


    # Q and target nural network
    self.nn = DeepNeuralNetwork(self.n_states, self.n_actions, self.learning_rate, len(nlp), nlp)
    self.nn_Q = self.nn.custom_network()
    self.nn_target = self.nn.custom_network()

    self.batch_size = train_max   # Size of batch taken from replay buffer
    self.num_episodes = 1000
    # self.max_steps_per_episode = 200
    # self.max_episodes = 10

    # Replay buffer stroing (state, action, reward, next_state, done) as doubly ended queue
    self.replay_buffer = deque(maxlen=2000)

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
    return random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
  
  def train(self):
    self.training_counts += 1

    batch_sample = self.sample_from_replay_memory()
    state_b = np.zeros((self.batch_size, self.n_states))
    next_state_b = np.zeros((self.batch_size, self.n_states))
    action_b, reward_b, done_b = [], [], []
    
    j=0
    for x in batch_sample:
      state_b[j] = x[0]
      next_state_b[j] = x[3]
      j+=1

      action_b.append(x[1])
      reward_b.append(x[2])
      done_b.append(x[4])

    target = self.nn_Q.predict(state_b)
    target_next = self.nn_target.predict(next_state_b)
  
    for i in range(self.batch_size):
      if done_b[i]:
        q_t = reward_b[i]
      else:
        q_t = reward_b[i] + self.gamma*np.amax(target_next[i])
      
      target[i][action_b[i]] = q_t

    result = self.nn_Q.fit(x=state_b, y=target, batch_size=self.batch_size, verbose=0)
    return result


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
        
        # if not done or i == self.env._max_episode_steps-1:
        #   reward = reward
        # else:
        #   reward = -reward # punishment for failure

        i+=1

        if self.RB_bool:
          self.remeber(state, action, reward, next_state, done)

        if len(self.replay_buffer) >= self.batch_size:
          result = self.train()
          loss.append(result.history['loss'])

        if self.TNN_bool and self.training_counts%self.weights_updating_frequancy == 0:
          self.nn_target.set_weights(self.nn_Q.get_weights())

        state = next_state
        score += reward
        steps.append(i)

      scores.append(score) 
      loss_avg.append(np.mean(loss))

      log = "Episode: {}/{}, Total reward: {}, Total steps: {}, Parameters: epsilon={}, lr={}".format(e, 
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


