import gymnasium as gym

class DQNAgent:
  def __init__(self,learning_rate,exploration_factor,policy):
    self.env = gym.make('CartPole-v1')
    self.learning_rate = learning_rate
    self.exploration_factor = exploration_factor
    self.policy == policy

  def select_action(self, s):
    ''' Selects next action using policy'''
    if self.policy == 'egreedy':
      epsilon = self.exploration_factor
      return
    if self.policy == 'softmax':
      return
 
  def burn_in_memory(self):
    pass

  def evaluate(self,eval_env,n_eval_episodes, max_episode_length=200):
    ''' Evaluates currently learned strategy '''
    return

  def train(self):
    return
