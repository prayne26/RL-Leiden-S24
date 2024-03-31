import gymnasium as gym
import numpy as np
import random
from collections import deque
from Helper import argmax, softmax
from Neural_network import DeepNeuralNetwork
import time

class DQNAgent:
  def __init__(self, learning_rate, gamma, policy, train_max, RB_bool=True, TNN_bool=True, temp=None, epsilon=None):
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


    # Enviorment
    self.env = gym.make('CartPole-v1')
    self.n_actions = self.env.action_space.n
    self.n_states = self.env.observation_space.shape[0]


    # Q and target nural network
    self.nn = DeepNeuralNetwork(self.n_states, self.n_actions, self.learning_rate)
    self.nn_Q = self.nn.custom_network()
    self.nn_target = self.nn.custom_network()

    self.batch_size = 32  # train_max # Size of batch taken from replay buffer
    self.num_episodes = 10000
    # self.max_steps_per_episode = 200
    # self.max_episodes = 10

    # Replay buffer stroing (state, action, reward, next_state, done) as doubly ended queue
    self.replay_buffer = deque(maxlen=2000)

  
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
  
  def plot_perf():
    pass
  
  def sample_from_replay_memory(self):
    return random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
  
  def train(self):
    self.training_counts += 1

    if self.TNN_bool:
      self.nn_target.set_weights(self.nn_Q.get_weights())

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

    results = self.nn_Q.fit(x=state_b, y=target, batch_size=self.batch_size, verbose=0)
    return results

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
        
        if not done or i == self.env._max_episode_steps-1:
          reward = reward
        else:
          reward = -reward # punishment for failure

        if self.RB_bool:
          self.remeber(state, action, reward, next_state, done)

        state = next_state
        score += reward
        i+=1

        if len(self.replay_buffer) >= self.train_max:
          results = self.train()
          loss.append(results.history['loss'])

        if done:
          steps.append(i)
          break
      
      scores.append(score) 
      loss_avg.append(np.mean(loss))

      print("Episode: {}, Total reward: {}, Total step: {}".format(e, score, steps[-1]))
    
    print('Scores: ', scores)
    print('Steps: ', steps)
    return loss_avg, steps




# def train(policy='egreedy',  exploration_factor=1, learning_rate=0.9):

#   # initialize environment, replay buffer, network & target network
#   agent = DQNAgent(learning_rate, exploration_factor, policy)
#   observation, info = agent.env.reset()
#   state = np.array(observation)

#   agent.initialize_replay_memory()
#   agent.initialize_Qnetwork()
#   agent.initialize_Qtnetwork()

#   # initial values
#   convergence = False
#   episode_reward =0

#   while convergence is False:
#     # select next action
#     action = agent.select_action(state)

#     # evaluate next state & reward
#     observation, reward, done, trunc = agent.execute_action(action)
#     next_state = np.array(observation)
#     episode_reward += reward

#     # update replay buffer
#     agent.store_to_replay_memory(action, state, next_state, reward, done)
#     state = next_state

#     # sample a batch from replay buffer (once there are enough entries for batch)
#     if len(agent.done_history) > agent.batch_size:
#       agent.sample_from_replay_memory()
#     agent.something_target_network()
#     agent.gradient_descent_Qnetwork()


#   agent.env.close()
#   return

# train()