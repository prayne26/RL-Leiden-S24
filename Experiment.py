from Agent import DQNAgent
import time 
import numpy as np

import matplotlib.pyplot as plt

def make_plots(perf, max_episodes, file_name, title):
    x = np.arange(1, max_episodes+1)
    j=0
    print(x, perf.values())
    plt.figure(figsize=(10, 6))
    for y in perf.values():
        j+=1
        plt.plot(x, y[0], label='DQN '+str(j))

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.legend(prop={'size':10})
    plt.grid(True)
    plt.savefig("Pics/"+file_name)
    plt.show()


def experiment():
    max_episodes = 300
    npl = [24,24]
    
    lr = 0.001
    gamma = 0.9,

    policy = 'egreedy'
    epsilon = 1.
    state_size = 4  
    action_size = 2  
    batch_size = 24

    agent = DQNAgent(state_size=state_size,
                        action_size=action_size,
                        learning_rate=lr,
                        gamma=gamma,
                        policy=policy,
                        batch_size=batch_size,
                        epsilon=epsilon,
                        npl=npl,
                        max_episodes=max_episodes)

    scores = agent.run()
    print(f'average score = {np.mean(scores)}')
    print(f'highest score = {np.max(scores)}')
    for interval_score in np.array_split(scores,10):
        print(f'   interval mean: {np.mean(interval_score)}')
    print(scores)
    #we should plot past 50 or 100 episode average instead of raw

def lr_experiment():
    max_episodes = 150
    npl = [24, 24]

    learning_rate = 0.001
    gamma = 0.9,

    policy = 'egreedy'
    epsilon = 0.99
    state_size = 4
    action_size = 2
    batch_size = 32

    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2]
    perf = {str(lr): [] for lr in learning_rates}
    perf_mean = {}
    for lr in learning_rates:
        # for _ in range(10):
        agent = DQNAgent(state_size=state_size,
                         action_size=action_size,
                         learning_rate=lr,
                         gamma=gamma,
                         policy=policy,
                         batch_size=batch_size,
                         epsilon=epsilon,
                         npl=npl,
                         max_episodes=max_episodes)

        scores = agent.run_experiment()
        perf[str(lr)].append(scores)
        # perf_mean.append(np.mean(scores))

    print("Averges after 10 runns of each config: ")
    for k, v in zip(perf.keys(), perf_mean):
        print("{} : {}+/-{}".format(k, np.mean(v[0]), np.std(v[0])))

    title = 'DQN Performance with Different NN architecture'
    file_name = 'Layers_perf.png'

    make_plot(perf, max_episodes, file_name, title)



def gamma_experiment():
    pass

def main():
    s = time.time()

    experiment()

    # Chacking various NN architecture
    #nn_experiment()
    
    # Checking various learning rate values
    #lr_experiment()

    # Checking various  values
    gamma_experiment()

    print("Program finished. Total time: {} seconds.".format(round(time.time()-s, 2)))

if __name__ == '__main__':
    main()
  