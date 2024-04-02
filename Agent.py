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
                 epsilon, tau, temp, NPL):
        '''npl - neurons per layer,, it will be a list with the numbers of neurons in the layers []'''
        self.n_states = state_size
        self.n_actions = action_size

        # adjustables
        self.policy = policy
        self.weights_updating_frequency = 30
        self.train_start = 500

        # hyperparameters
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # polyak coefficient for updating target model. if None, uses total replacement every _ training steps
        self.temp = temp

        # fixed parameters
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.initial_epsilon = epsilon  # for epsilon reset
        self.max_steps = 500  # the envirment limit
        self.replay_buffer = deque(maxlen=2000)

        # neural network setup
        if NPL is None:
            NPL = [24, 24]
        self.model_Q = self.custom_network(NPL)  # main neural network
        self.model_T = self.custom_network(NPL)  # target neural network
        self.update_target_model(None)

        self.total_step_count = 0

    def custom_network(self, NPL):
        # if len(self.neurons_per_layer) != self.n_layers:
        #     print("Wrong number! More/less elements in the list neurons_per_layer then the layers number.")
        #     sys.exit()
        #     return
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.n_states,)))
        for l in range(len(NPL)):
            model.add(layers.Dense(NPL[l], activation='relu', name='L' + str(l)))
        model.add(layers.Dense(self.n_actions, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

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
                a = np.argmax(self.model_Q.predict(state, verbose=0))

        elif self.policy == 'softmax':
            if self.temp is None:
                raise KeyError("Provide a temperature")
            a = np.random.choice(range(self.n_actions), 1,
                                 p=softmax(self.model_Q.predict(state, verbose=0), self.temp))[0]
        else:
            raise KeyError('Provide a valid policy')
        return a

    def replay(self):
        if len(self.replay_buffer) < self.train_start:
            return
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = self.model_Q.predict(state, verbose=0)
            q_action = np.argmax(self.model_Q.predict(next_state, verbose=0)[0])
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

        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset_epsilon(self):
        self.epsilon = self.initial_epsilon

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
                max_episodes=200):
    # starting run
    env = gym.make('CartPole-v1')
    state_size, action_size = env.observation_space.shape[0], env.action_space.n
    print(f'statesize:{state_size}, actionsize={action_size}')
    agent = DQNAgent(state_size, action_size, batch_size, policy, learning_rate, gamma, epsilon, tau, temp, NPL)
    agent.clear_log()
    scores, evals = [], []

    for e in range(max_episodes):  # we may try diffrent criterion for stopping
        state, _ = env.reset(seed=0)
        state = np.reshape(state, [1, agent.n_states])
        for step in range(agent.max_steps):  # CartPole-v1 enforced max step
            action = agent.act(state)
            next_state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            next_state = np.reshape(next_state, [1, agent.n_states])
            reward = reward if not done else -100
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state

            if done:
                scores.append(step)
                train = True if len(agent.replay_buffer) > agent.train_start else False

                log = "Episode: {}/{}, Train steps: {}, train:{}, Parameters: epsilon={}, lr={}.".format(
                    e + 1, max_episodes, step, train, agent.epsilon, agent.learning_rate)
                print(log)
                if agent.tau is not None:
                    agent.update_target_model(agent.tau)
                    if train and e%5 == 0:
                        evalT = agent.evaluate(env, 'T')
                        evals.append(evalT)
                        print(f'Eval = {evalT}')


                agent.reset_epsilon()
                # agent.save_log(log)
                break

        if agent.tau is None:
            if agent.total_step_count % agent.weights_updating_frequency == 0:
                agent.update_target_model(None)
                evalT = agent.evaluate(env, 'T')
                evals.append(evalT)
                print(f'Eval = {evalT}')

    env.close()
    return scores, evals

    #
    # def run_experiment(self):
    #     env = gym.make('CartPole-v1')
    #     scores = []
    #     for e in range(self.max_episodes):
    #         state, _ = env.reset()
    #         state = np.reshape(state, [1, self.n_states])
    #         score = 0
    #         for step in range(self.max_steps):
    #             action = self.act(state)
    #             next_state, reward, done, info, _ = env.step(action)
    #             next_state = np.reshape(next_state, [1, self.n_states])
    #
    #             self.remember(state, action, reward, next_state, done)
    #
    #             state = next_state
    #             score += reward
    #             if done:
    #                 break
    #
    #         if len(self.replay_buffer) > self.batch_size:
    #             self.replay()
    #
    #         if step % self.weights_updating_frequency == 0:
    #             self.update_target_model()
    #
    #         scores.append(score)
    #         print("Episode: {}/{}, Score: {}".format(e + 1, self.max_episodes, score))
    #
    #     return scores
