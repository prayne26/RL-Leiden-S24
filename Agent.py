import gymnasium as gym
import numpy as np
import os
import random
from collections import deque
from Helper import argmax, softmax
from Neural_network import DeepNeuralNetwork


class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, policy, learning_rate, gamma, epsilon, npl, max_episodes,
                 solved=False):
        '''npl - neurons per layer,, it will be a list with the numbers of neurons in the layers []'''
        self.n_state = state_size
        self.n_actions = action_size
        self.replay_buffer = deque(maxlen=2000)

        self.policy = policy
        self.gamma = float(gamma[0])  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.initial_epsilon = epsilon #for epsilon reset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = None  # polyak coefficient for updating target model. if None, uses total replacement every _ training steps

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.train_start = 200

        nn = DeepNeuralNetwork(state_size, action_size, learning_rate, len(npl), npl)
        self.model_Q = nn.custom_network()  # main neural network
        self.model_T = nn.custom_network()  # target neural network
        self.update_target_model()

        self.max_episodes = max_episodes
        self.max_steps = 500  # the envirment limit
        self.weights_updating_frequency = 30
        self.total_step_count = 0

    def update_target_model(self, tau=None):
        # Copy weights from the main model to target_model
        if tau is None:
            self.model_T.set_weights(self.model_Q.get_weights())
        else:
            # Polyak averaging
            current_weights = self.model_Q.get_weights()
            target_weights = self.model_T.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = current_weights[i] * tau + target_weights[i] * (1 - tau)
            self.model_T.set_weights(target_weights)

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
                a = argmax(self.model_Q.predict(state, verbose=0))

        elif self.policy == 'softmax':
            if self.temp is None:
                raise KeyError("Provide a temperature")

            a = \
                np.random.choice(range(self.n_actions), 1,
                                 p=softmax(self.model_Q.predict(state, verbose=0), self.temp))[0]

        return a


    def replay(self):
        if len(self.replay_buffer) > self.train_start:
            return
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = self.model_Q.predict(state, verbose=0)
            q_action = np.amax(self.model_Q.predict(next_state, verbose=0)[0])
            next_qval = self.model_T.predict(next_state, verbose=0)[0][q_action]
            # next_qval = np.amax(self.model_T.predict(next_state, verbose=0)[0])
            target[0][action] = reward
            if not done:
                target[0][action] = reward + self.gamma * next_qval
            if self.tau is None:
                # First method, updating target network every n timesteps by total replacement
                self.model_Q.fit(state, target, epochs=1, verbose=0)
                self.total_step_count += 1
            else:
                # Second method, updating target network in batches and target every episode
                states.append(state[0])
                targets.append(target[0])

        if self.tau is not None:
            self.model_Q.fit(np.array(states), np.array(targets), batch_size=self.batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model_Q.load_weights(name)

    def save(self, name):
        self.model_Q.save_weights(name)

    def clear_log(self):
        path = "Logs/"
        file_name = "log.txt"
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            with open(path + file_name, "w") as myfile:
                myfile.write("")
        except:
            print("Unable to clear the file.")

    def save_log(self, log):
        path = "Logs/"
        file_name = "log.txt"
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            with open(path + file_name, "a") as myfile:
                myfile.write(log)
        except:
            print("Unable to save the file.")

    def run(self):
        print("Starting running...")
        self.clear_log()
        env = gym.make('CartPole-v1')
        scores = deque(maxlen=100)
        for e in range(self.max_episodes):  # we may try diffrent criterion for stopping
            state, _ = env.reset(seed=0)
            state = np.reshape(state, [1, self.n_state])
            self.epsilon = self.initial_epsilon

            for step in range(self.max_steps):  # CartPole-v1 enforced max step
                action = self.act(state)
                next_state, reward, done, info, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.n_state])

                #lower final reward if terminated
                reward = reward if not done else -100

                self.remember(state, action, reward, next_state, done)

                self.replay()

                if done:
                    scores.append(step)
                    if self.tau is not None:
                        self.update_target_model(self.tau)

                    log = "Episode: {}/{}, Total steps: {}, Parameters: epsilon={}, lr={}.\n".format(
                        e + 1, self.max_episodes, step, self.epsilon, self.learning_rate)
                    print(log)
                    self.save_log(log)
                    break
                state = next_state

            if self.tau is None:
                if self.total_step_count % self.weights_updating_frequency == 0:
                    self.update_target_model(self.tau)

            #print("Episode: {}/{}, Score: {}".format(e + 1, self.max_episodes, rewards))
        env.close()
        return scores

    def run_experiment(self):
        env = gym.make('CartPole-v1')
        scores = []
        for e in range(self.max_episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.n_state])
            score = 0
            for step in range(self.max_steps):
                action = self.act(state)
                next_state, reward, done, info, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.n_state])

                self.remember(state, action, reward, next_state, done)

                state = next_state
                score += reward
                if done:
                    break

            if len(self.replay_buffer) > self.batch_size:
                self.replay()

            if step % self.weights_updating_frequency == 0:
                self.update_target_model()

            scores.append(score)
            print("Episode: {}/{}, Score: {}".format(e + 1, self.max_episodes, score))

        return scores
