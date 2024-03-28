import gymnasium as gym
import time
import numpy as np
from numpy.random import choice
from Helper import argmax
from Neural_network import DeepNeuralNetwork
import keras
import tensorflow


# @Bartosz, many of the def() functions are probably redundant and can be coded straight into
#       the final train() function. I just added them as placeholders for the algorithm.

class DQNAgent:
    def __init__(self, learning_rate, exploration_factor, policy):
        self.env = gym.make('CartPole-v1')
        self.learning_rate = learning_rate
        self.discount_factor = 0.99
        self.exploration_factor = exploration_factor
        self.policy = policy

        self.n_actions = 2
        self.max_steps_per_episode = 200
        self.max_episodes = 10

        self.batch_size = 32  # Size of batch taken from replay buffer

        self.Qnetwork = None
        self.QTnetwork = None
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
        print(
            f'self.histories = {self.histories},shape={np.asarray(self.histories).shape} type hist[0]={type(self.histories[0])}')

    def select_action(self, s):
        ''' Selects next action using initialized policy'''
        if self.policy == 'egreedy':
            probabilities = []
            epsilon = self.exploration_factor

            # extracting Q values from Qnetwork
            print(f'state={s}')
            state_tensor = keras.ops.convert_to_tensor(s)
            print(f'state_tensor1={state_tensor},type={type(state_tensor)}')
            state_tensor = keras.ops.expand_dims(state_tensor, 0)
            print(f'state_tensor2={state_tensor},type={type(state_tensor)}')
            action_probs = self.Qnetwork(state_tensor, training=False)
            print(f'action_probs = {action_probs}')

            # epsilon greedy algorithm
            max_a = argmax(action_probs[0])
            print(f'max_a = {max_a}')
            for a in range(self.n_actions):
                if a == max_a:
                    probability = 1. - epsilon * (self.n_actions - 1) / self.n_actions
                else:
                    probability = epsilon / self.n_actions
                probabilities.append(probability)
            chosen_a = choice(range(self.n_actions), 1, p=probabilities)
            print(f'chosen_a = {chosen_a}')
            return int(chosen_a)

        if self.policy == 'softmax':
            return

        else:
            print('please specify a policy')

    def execute_action(self, a):
        observation, reward, terminated, truncated, info = self.env.step(a)
        return observation, reward, terminated, truncated

    def sample_from_replay_memory(self):
        if len(self.histories) > self.batch_size:
            sample = np.random.choice(self.histories, size=self.batch_size)
        else:
            sample = self.histories
        print(f'sample length= {len(sample)}')
        print(f'first entry={sample[0]}')
        return sample

    def update_networks(self, i, batch):
        target_update_interval = 50

        for action, state, next_state, reward, done in batch:
            G = reward
            if not done:
                G = reward + self.discount_factor * np.amax(self.QTnetwork.predict(next_state)[0])

            prediction = self.QTnetwork.predict(state)
            prediction[0][action] = G
            self.Qnetwork.fit(state, prediction, epochs=1, verbose=0)
        if i % target_update_interval == 0:
            self.QTnetwork.set_weights(self.Qnetwork.get_weights())

    def evaluate(self, eval_env, n_eval_episodes, max_episode_length=200):
        ''' Evaluates currently learned strategy '''
        return

    def train(self):
        learning_rate = 0.9
        exploration_factor = 1
        policy = 'egreedy'

        # initialize environment, replay buffer, network & target network
        observation, info = self.env.reset()
        state = np.array(observation)

        self.initialize_replay_memory()
        self.Qnetwork = DeepNeuralNetwork.custom_Qnetwork
        self.QTnetwork = DeepNeuralNetwork.custom_Qnetwork

        # initial values
        episode_reward = 0

        for i in range(5):
            # select next action
            action = self.select_action(state)

            # evaluate next state & reward
            next_state, reward, done, trunc = self.execute_action(action)
            episode_reward += reward

            # update replay buffer
            print(f'appending {[action, state, next_state, reward, done]}')
            print(type(action))
            print(type(state))
            print(type(next_state))
            print(type(reward))
            print(type(done))
            self.histories.append([action, state, next_state, reward, done])
            state = next_state

            # sample a batch from replay buffer (once there are enough entries for batch)
            batch = self.sample_from_replay_memory()

            # use batch to train Qnetwork, trains QTnetwork every 50 iterations
            self.update_networks(i, batch)
        self.env.close()
        return


DQNAgent.train()
