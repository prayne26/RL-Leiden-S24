import gymnasium as gym
import time
import numpy as np
from numpy.random import choice
from Helper import argmax
from Neural_network import DeepNeuralNetwork
import keras
import tensorflow



class DQNAgent:
    def __init__(self, learning_rate=0.001, exploration_factor=0.1, policy='egreedy', verbose=0):
        self.env = gym.make('CartPole-v1')
        self.learning_rate = learning_rate
        self.discount_factor = 0.99
        self.exploration_factor = exploration_factor
        self.policy = policy
        self.verbose = verbose

        self.n_actions = 2
        self.max_steps_per_episode = 200
        self.max_episodes = 10

        self.batch_size = 32  # Size of batch taken from replay buffer

        self.Q_network = None
        self.QT_network = None
        self.histories = []  # replay memory
        self.running_reward = 0
        self.episode_count = 0

    def initialize_replay_memory(self):
        # probably redundant
        action_history = []
        state_history = []
        next_state_history = []
        reward_history = []
        done_history = []
        self.histories = [[action_history, state_history, next_state_history, reward_history, done_history]]

    def select_action(self, s):
        ''' Selects next action using initialized policy'''
        if self.policy == 'egreedy':
            probabilities = []
            epsilon = self.exploration_factor

            # extracting Q values from Qnetwork
            s = np.expand_dims(s, axis=(0, 1))
            action_probs = self.Q_network.predict(s, verbose=0)

            # epsilon greedy algorithm
            max_a = argmax(action_probs[0])
            for a in range(self.n_actions):
                if a == max_a:
                    probability = 1. - epsilon * (self.n_actions - 1) / self.n_actions
                else:
                    probability = epsilon / self.n_actions
                probabilities.append(probability)
            chosen_a = choice(range(self.n_actions), 1, p=probabilities)

            if self.verbose:
                print(f'    state={s}')
                print(f'    action_probs = {action_probs}')
                print(f'    max_a = {max_a}, chosen_a = {chosen_a}')
            return int(chosen_a)
        if self.policy == 'softmax':
            return
        else: print('please specify a policy:{egreedy, softmax}')

    def execute_action(self, a):
        observation, reward, terminated, truncated, info = self.env.step(a)
        return observation, reward, terminated, truncated

    def sample_batch(self, ):
        '''Samples a batch of size {self.batch_size}, from replay history/memory '''
        if len(self.histories) > self.batch_size:
            index_samples = np.random.choice(len(self.histories), size=self.batch_size, replace=False)
            sample = [self.histories[i] for i in index_samples]
        else:
            sample = self.histories[1:]
        if self.verbose: print(f'    sample length= {len(sample)},   first batch item ={sample[0]}')
        return sample

    def update_networks(self, i, batch):
        # not 100% confident i have this set up correctly
        # could be a mixup in QT_network and Q_network in network.predict(s)
        target_update_interval = 25 #50
        for action, state, next_state, reward, done in batch:
            if i > 30:
                print(f'{i}: {np.shape(batch)}'
                      f'next state size = {np.shape(next_state)}: {next_state}'
                      f'{batch}')
            next_state = np.expand_dims(next_state, axis=(0, 1))
            state = np.expand_dims(state, axis=(0, 1))
            G = reward
            if not done:
                target_predictions = self.QT_network.predict(next_state, verbose=0)
                G = reward + self.discount_factor * np.amax(target_predictions[0])

            prediction = self.QT_network.predict(state, verbose=0)
            prediction[0][action] = G
            self.Q_network.fit(state, prediction, epochs=1, verbose=0)
        if i % target_update_interval == 0:
            self.QT_network.set_weights(self.Q_network.get_weights())

    def evaluate(self, eval_env, n_eval_episodes, max_episode_length=200):
        ''' Evaluates currently learned strategy '''
        return

    def train(self, nr_layers=1, neurons_per_layer=16):
        # initialize neural network & target network
        DNN = DeepNeuralNetwork(nr_layers=nr_layers, neurons_per_layer=neurons_per_layer,
                                learning_rate=self.learning_rate)
        self.Q_network = DNN.custom_Qnetwork()
        self.QT_network = DNN.custom_Qnetwork()

        # initial values
        for i in range(self.max_episodes):
            # initialize environment & replay buffer
            self.initialize_replay_memory()
            observation, info = self.env.reset()
            state = np.array(observation)

            episode_reward = 0
            for j in range(self.max_steps_per_episode):
                if self.verbose: print(f'\nepisode {i}: iteration {j}')
                # select next action
                action = self.select_action(state)

                # evaluate next state & reward
                next_state, reward, done, trunc = self.execute_action(action)
                episode_reward += reward

                # update replay buffer
                self.histories.append([action, state, next_state, reward, done])
                # action ->   <class 'int'>
                # state ->    <class 'numpy.ndarray'> size (4,)
                # next_state> <class 'numpy.ndarray'> size (4,)
                # reward ->   <class 'float'>
                # done ->     <class 'bool'>
                state = next_state

                # sample a batch from replay buffer (once there are enough entries for batch)
                batch = self.sample_batch()

                # use batch to train Qnetwork, trains QT_network every 50 iterations
                self.update_networks(j, batch)

                if done:
                    print(f'episode {i} terminated at iteration {j} - episode reward = {episode_reward}')
                    break
        self.env.close()
        return


agent = DQNAgent(verbose=0)
agent.train()
