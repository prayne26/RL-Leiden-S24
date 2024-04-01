from Agent import DQNAgent
import time 
import numpy as np


import matplotlib.pyplot as plt

def plot_perf(loss_avg, stepst):
    pass

def nn_experiment_plot(loss, max_episodes):
    x = np.arange(0, max_episodes)

    plt.figure(figsize=(10, 6))
    i = 0
    for y in loss:
        i+=1
        plt.plot(x, y, '-', label='NN '+str(i))
    plt.ylabel('Episodes', fontsize=12)
    plt.xlabel('Loss function', tontsize=12)
    plt.title('Loss function in different architecture', fontsize=14)
    plt.legend(prop={'size': 10})
    plt.savefig("Loss_arch.png")
    plt.show()
def nn_experiment():
    max_episodes = 201
    npls = [[32], [32,32], [32,32,32]]
    
    learning_rate = 0.001
    gamma = 0.95, 

    policy = 'egreedy'
    epsilon = 0.5
    state_size = 4  
    action_size = 2  
    batch_size = 32 

    loss_nn, rewards_nn = [], []
    for npl in npls:
        agent = DQNAgent(state_size=state_size,
                        action_size=action_size,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        policy=policy,
                        batch_size=batch_size,
                        epsilon=epsilon,
                        npl=npl,
                        max_episodes=max_episodes)
    
        loss_avg, tot_rewards = agent.run()
        loss_nn.append(loss_avg)
        rewards_nn.append(tot_rewards)


    nn_experiment_plot(loss_nn, max_episodes)


def main():
    s = time.time()
    nn_experiment()
    
    print("Program finished. Total time: {} seconds.".format(round(time.time()-s, 2)))

if __name__ == '__main__':
    main()
  