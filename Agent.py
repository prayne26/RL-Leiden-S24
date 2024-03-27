import gymnasium as gym
import time
from numpy.random import choice
from Helper import argmax


class DQNAgent:
  def __init__(self, learning_rate, exploration_factor, policy):
    self.env = gym.make('CartPole-v1')
    self.learning_rate = learning_rate
    self.exploration_factor = exploration_factor
    self.policy = policy

  def initialize_replay_memory(self):
    pass

  def initialize_Qnetwork(self):
    pass

  def initialize_Qtnetwork(self):
    pass

  def select_action(self, s):
    ''' Selects next action using policy'''
    if self.policy == 'egreedy':
      ''' Returns the epsilon-greedy best action in state s '''
      probabilities = []
      epsilon = self.exploration_factor
      # collect probability distribution of the possible actions
      max_a = argmax(self.Q_sa[s, :])  # setting argmax in advance to prevent multiplicity of if statement
      for a in range(self.n_actions):
        if a == max_a:
          probability = 1. - epsilon * (self.n_actions - 1) / self.n_actions
        else:
          probability = epsilon / self.n_actions
        probabilities.append(probability)
      # using choice from numpy.random to select action according to probability distribution
      chosen_a = choice(range(self.n_actions), 1, p=probabilities)
      return int(chosen_a)

      return
    if self.policy == 'softmax':
      return
    else:
      print('please specify a policy')

  def execute_action(self,a):
    observation, reward, terminated, truncated, info = self.env.step(a)
    return observation, reward, terminated, truncated

  def store_to_replay_memory(self):
    pass


  def sample_from_replay_memory(self):
    pass


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
  agent = DQNAgent(learning_rate, exploration_factor, policy)
  agent.initialize_replay_memory()
  agent.initialize_Qnetwork()
  agent.initialize_Qtnetwork()
  convergence = False
  s,info = agent.env.reset()
  while convergence is False:
    a = agent.select_action(s)
    agent.execute_action(a)
    agent.store_to_replay_memory()
    agent.sample_from_replay_memory()
    agent.something_target_network()
    agent.gradient_descent_Qnetwork()


  agent.env.close()
  return

train()