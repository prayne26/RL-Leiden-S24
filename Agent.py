import gymnasium as gym
import time
import numpy as np
from numpy.random import choice
from Helper import argmax
from Neural_network import DeepNeuralNetwork
import keras

# @Bartosz, many of the def() functions are probably redundant and can be coded straight into
#       the final train() function. I just added them as placeholders for the algorithm.

class DQNAgent:
  def __init__(self, learning_rate, exploration_factor, policy):
    self.env = gym.make('CartPole-v1')
    self.learning_rate = learning_rate
    self.exploration_factor = exploration_factor
    self.policy = policy

    self.n_actions = 2

    self.batch_size = 32  # Size of batch taken from replay buffer
    self.max_steps_per_episode = 200
    self.max_episodes = 10

    # replay buffer variables
    self.action_history = []
    self.state_history = []
    self.next_state_history = []
    self.reward_history = []
    self.done_history = []
    self.episode_reward_history = []
    self.running_reward = 0
    self.episode_count = 0

  def initialize_replay_memory(self):
    # probably redundant
    self.action_history = []
    self.state_history = []
    self.next_state_history = []
    self.reward_history = []
    self.done_history = []
    pass

  def initialize_Qnetwork(self):
    self.Qnetwork = DeepNeuralNetwork.custom_Qnetwork()
    pass

  def initialize_Qtnetwork(self):
    self.QTnetwork = DeepNeuralNetwork.custom_Qnetwork()
    pass

  def select_action(self, s):
    ''' Selects next action using policy'''
    if self.policy == 'egreedy':
      ''' Returns the epsilon-greedy best action in state s '''
      probabilities = []
      epsilon = self.exploration_factor

      #extracting Q values from Qnetwork
      state_tensor = keras.ops.convert_to_tensor(s)
      state_tensor = keras.ops.expand_dims(state_tensor, 0)
      action_probs = self.Qnetwork(state_tensor, training=False)

      #epsilon greedy algorithm
      max_a = argmax(action_probs[0])
      for a in range(self.n_actions):
        if a == max_a:
          probability = 1. - epsilon * (self.n_actions - 1) / self.n_actions
        else:
          probability = epsilon / self.n_actions
        probabilities.append(probability)
      chosen_a = choice(range(self.n_actions), 1, p=probabilities)
      return int(chosen_a)


    if self.policy == 'softmax':
      return

    else:
      print('please specify a policy')

  def execute_action(self,a):
    observation, reward, terminated, truncated, info = self.env.step(a)
    return observation, reward, terminated, truncated

  def store_to_replay_memory(self, action, state, next_state, reward, done):
    self.action_history.append(action)
    self.state_history.append(state)
    self.next_state_history.append(next_state)
    self.reward_history.append(reward)
    self.done_history.append(done)
    pass

  def sample_from_replay_memory(self):
    indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

    # Using list comprehension to sample from replay buffer
    state_sample = np.array([self.state_history[i] for i in indices])
    next_state_sample = np.array([self.next_state_history[i] for i in indices])
    reward_sample = [self.reward_history[i] for i in indices]
    action_sample = [self.action_history[i] for i in indices]
    done_sample = [self.done_history[i] for i in indices]
    return state_sample, next_state_sample, reward_sample, action_sample, done_sample


  def something_target_network(self):
    pass


  def gradient_descent_Qnetwork(self):
    pass

  def evaluate(self,eval_env,n_eval_episodes, max_episode_length=200):
    ''' Evaluates currently learned strategy '''
    return




def train():
  learning_rate = 0.9
  exploration_factor = 1
  policy = 'egreedy'

  # initialize environment, replay buffer, network & target network
  agent = DQNAgent(learning_rate, exploration_factor, policy)
  observation, info = agent.env.reset()
  state = np.array(observation)

  agent.initialize_replay_memory()
  agent.initialize_Qnetwork()
  agent.initialize_Qtnetwork()

  # initial values
  convergence = False
  episode_reward =0

  while convergence is False:
    # select next action
    action = agent.select_action(state)

    # evaluate next state & reward
    observation, reward, done, trunc = agent.execute_action(action)
    next_state = np.array(observation)
    episode_reward += reward

    # update replay buffer
    agent.store_to_replay_memory(action, state, next_state, reward, done)
    state = next_state

    # sample a batch from replay buffer (once there are enough entries for batch)
    if len(agent.done_history) > agent.batch_size:
      agent.sample_from_replay_memory()
    agent.something_target_network()
    agent.gradient_descent_Qnetwork()


  agent.env.close()
  return

train()