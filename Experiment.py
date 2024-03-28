import gymnasium as gym
import numpy as np
import time

def previewEnvironment():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset()
    print(f'observation_space={env.observation_space}')
    print(f'action space={env.action_space}')
    shape1 = (1,)+env.observation_space.shape
    shape2 = (1,list(env.observation_space.shape)[0])
    print(shape1)
    print(np.random.randint(0,3,size=shape1))
    print(shape2)
    print(np.random.randint(0, 3, size=shape2))
    print(f'initial state = {observation}')
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.1)
        print(f'next state = {observation}, reward = {reward}')

        if terminated or truncated:
            observation, info = env.reset()
            print('kil\n')
    env.close()


import pygame
from gymnasium.utils.play import play
def play_cartgame():
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
    play(gym.make("CartPole-v1", render_mode='rgb_array'), keys_to_action=mapping, fps=5)

previewEnvironment()
