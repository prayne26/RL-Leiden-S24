import gymnasium as gym
import numpy as np
import os
import random
from collections import deque
from Helper import softmax
from keras import layers
import keras
from keras.optimizers import Adam
import sys


class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, policy, learning_rate, gamma,
                 epsilon, tau, temp, NPL, ddqn):
        '''npl - neurons per layer,, it will be a list with the numbers of neurons in the layers []'''
        self.n_states = state_size
        self.n_actions = action_size

        # adjustables
        self.policy = policy
        self.weights_updating_frequency = 20
        self.train_start = 1000
        self.ddqn = ddqn

        # hyperparameters
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # polyak coefficient for updating target model. if None, uses total replacement every _ training steps
        self.temp = temp

        # fixed parameters
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.max_steps = 200  # the envirment limit
        self.replay_buffer = deque(maxlen=2000)

        # neural network setup
        if NPL is None:
            NPL = [24, 24]
        self.model_Q = self.custom_network(NPL)  # main neural network
        self.model_T = self.custom_network(NPL)  # target neural network
        self.model_T.set_weights(self.model_Q.get_weights())

        self.total_step_count = 0

    def custom_network(self, NPL):
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.n_states,)))
        for l in range(len(NPL)):
            model.add(layers.Dense(NPL[l], activation='relu', name='L' + str(l)))
        model.add(layers.Dense(self.n_actions, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self, tau=None, pa_toggle=False):
        # Copy weights from the main model to target_model
        if pa_toggle:
            # Polyak averaging
            current_weights = self.model_Q.get_weights()
            target_weights = self.model_T.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = current_weights[i] * tau + target_weights[i] * (1 - tau)
            self.model_T.set_weights(target_weights)
        else:
            self.model_T.set_weights(self.model_Q.get_weights())

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
                a = np.argmax(self.model_Q.predict(state, verbose=0))

        elif self.policy == 'softmax':
            if self.temp is None:
                raise KeyError("Provide a temperature")
            a = np.random.choice(range(self.n_actions), 1,
                                 p=softmax(self.model_Q.predict(state, verbose=0), self.temp))[0]
        else:
            raise KeyError('Provide a valid policy')
        return a

    def replay(self, no_tn, no_er):
        if len(self.replay_buffer) < self.train_start:
            return
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            if not no_tn:
                self.model_T.set_weights(self.model_Q.get_weights())

            if self.ddqn:
                q_action = np.argmax(self.model_Q.predict(next_state, verbose=0)[0])
                next_qval = self.model_T.predict(next_state, verbose=0)[0][q_action]
            else:
                next_qval = np.amax(self.model_T.predict(next_state, verbose=0)[0])

            target = self.model_Q.predict(state, verbose=0)
            target[0][action] = reward
            if not done:
                target[0][action] = reward + self.gamma * next_qval
            states.append(state[0])
            targets.append(target[0])

        self.total_step_count += 1
        self.model_Q.fit(np.array(states), np.array(targets), batch_size=self.batch_size, epochs=1, verbose=0)

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            if self.epsilon_decay < 1:
                print(f'min epsilon={self.epsilon} reached')
            self.epsilon_decay = 1.
        
        if no_er: 
            self.replay_buffer.clear()

    def load(self, name):
        self.model_Q.load_weights(name)

    def save(self, name):
        self.model_Q.save_weights(name)

    def evaluate(self, env, model):
        state, _ = env.reset(seed=0)
        state = np.reshape(state, [1, self.n_states])
        for step in range(200):
            if model == 'T':
                action = np.argmax(self.model_T.predict(state, verbose=0))
            else:
                action = np.argmax(self.model_Q.predict(state, verbose=0))
            next_state, reward, term, trunc, info = env.step(action)
            next_state = np.reshape(next_state, [1, self.n_states])
            done = term or trunc
            if done:
                return step
            state = next_state
        return 200


def dqn_learner(batch_size=24,
                policy='egreedy',
                learning_rate=0.001,
                gamma=0.9,
                epsilon=1.,
                tau=0.1,
                temp=0.1,
                NPL=None,
                max_episodes=200,
                ddqn=False,
                no_ER=False,
                no_TN=False):
    # starting run
    env = gym.make('CartPole-v1')
    state_size, action_size = env.observation_space.shape[0], env.action_space.n
    agent = DQNAgent(state_size, action_size, batch_size, policy, learning_rate, gamma, epsilon, tau, temp, NPL, ddqn)
    #agent.clear_log()
    scores, evals = [], []

    for e in range(max_episodes):
        state, _ = env.reset(seed=0)
        state = np.reshape(state, [1, agent.n_states])
        for step in range(agent.max_steps):  # CartPole-v1 enforced max step
            action = agent.act(state)
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            next_state = np.reshape(next_state, [1, agent.n_states])
            reward = reward if not done else -100
           
            agent.remember(state, action, reward, next_state, done)     
            agent.replay(no_TN, no_ER)

            state = next_state

            if done:
                scores.append(step)
                train = True if len(agent.replay_buffer) > agent.train_start else False

                log = "Episode: {}/{}, score: {}, train:{}, training count: {}".format(
                    e + 1, max_episodes, step, train, agent.total_step_count)
                print(log)
                agent.update_target_model(agent.tau, pa_toggle=True)
                break
            # if agent.total_step_count%agent.weights_updating_frequency==0 and agent.total_step_count != 0 and not ddqn:
            #     agent.update_target_model()


        if len(agent.replay_buffer) > agent.train_start and e % 5 == 0:
            evalT = agent.evaluate(env, 'T')
            evals.append(evalT)
            print(f'Eval T = {evalT}')

    env.close()
    return scores, evals