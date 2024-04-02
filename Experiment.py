from Agent import DQNAgent, dqn_learner
import time
import numpy as np
import os
import sys
import pickle

import matplotlib.pyplot as plt


def make_plot(perf, max_episodes, file_name, title):
    x = np.arange(1, max_episodes + 1)
    j = 0
    print(x, perf.values())
    plt.figure(figsize=(10, 6))
    for y in perf.values():
        j += 1
        plt.plot(x, y[0], label='DQN ' + str(j))

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(range(0,  200, 20))
    plt.title(title, fontsize=14)

    plt.legend(prop={'size': 10})
    plt.grid(True)
    plt.savefig("Pics/" + file_name)
    plt.show()


def save_run(scores, evals, general_title, test_title):
    path = "Logs/"
    file_name = general_title + '_' + test_title + ".txt"
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        # pickle to store lists instead of strings
        with open(path + file_name, "wb") as myfile:
            pickle.dump([scores, evals], myfile)
    except:
        print("Unable to save the file.")


def load_run(general_title, test_title):
    path = "Logs/"
    file_name = general_title + '_' + test_title + ".txt"
    try:
        # pickle to store lists instead of strings
        with open(path + file_name, "rb") as myfile:
            return pickle.load(myfile)
    except:
        print("Unable to read the file.")

def dqn_vs_ddqn():
    general_title = 'dqnVSddqn'
    max_episodes = 200

    test_title = 'dqn'
    scores, evals = dqn_learner(ddqn=False,max_episodes=max_episodes)
    save_run(scores, evals, general_title, test_title)

    test_title = 'ddqn'
    scores, evals = dqn_learner(ddqn=False, max_episodes=max_episodes)
    save_run(scores, evals, general_title, test_title)

def nn_experiment():
    npls1 = [12], [12, 12], [12, 12, 12, 12]
    npls2 = [8], [24], [128]
    general_title = 'nn_experiment'
    max_episodes = 200
    for npl in npls1:
        test_title = str(npl)
        scores, evals = dqn_learner(NPL=npl, max_episodes=max_episodes)
        save_run(scores, evals, general_title, test_title)

    for npl in npls2:
        test_title = str(npl)
        scores, evals = dqn_learner(NPL=npl, max_episodes=max_episodes)
        save_run(scores, evals, general_title, test_title)


def lr_experiment():
    max_episodes = 100
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

        scores, evals = dqn_learner(NPL=npl, max_episodes=max_episodes)

        perf[str(lr)].append(scores)
        # perf_mean.append(np.mean(scores))

    print("Averges after 10 runns of each config: ")
    for k, v in zip(perf.keys(), perf_mean):
        print("{} : {}+/-{}".format(k, np.mean(v[0]), np.std(v[0])))

    title = 'DQN Performance with different NN architecture'
    file_name = 'Layers_perf.png'

    make_plot(perf, max_episodes, file_name, title)


def gamma_experiment():
    general_title = 'nn_experiment'
    max_episodes = 150
    policies = ['egreedy', 'softmax']
    for policy in policies:
        test_title = str(policy)
        scores, evals = dqn_learner(policy=policy,max_episodes=max_episodes)
        save_run(scores, evals, general_title, test_title)


def ablation_study(no_er, no_tn):
    if no_er == False and no_tn == False:
        print("Error! Comparing DQN with DQN. It works don't worry about it :)")
        return
    
    max_episodes = 100
    npl = [32,32]
    
    learning_rate = 0.001 # Here should be tuned value
    gamma = 0.95          # Here should be tuned value

    policy = 'egreedy'
    epsilon = 0.8
    state_size = 4  
    action_size = 2  
    batch_size = 32 
    
    scores_dqn, evals_dqn = dqn_learner(state_size=state_size,
                                        action_size=action_size,
                                        learning_rate=learning_rate,
                                        gamma=gamma,
                                        policy=policy,
                                        batch_size=batch_size,
                                        epsilon=epsilon,
                                        NPL=npl,
                                        tau=0.1,
                                        max_episodes=max_episodes)
    
    scores_comp, evals_comp = dqn_learner(state_size=state_size,
                                          action_size=action_size,
                                          learning_rate=learning_rate,
                                          gamma=gamma,
                                          policy=policy,
                                          batch_size=batch_size,
                                          epsilon=epsilon,
                                          NPL=npl,
                                          tau=0.1,
                                          max_episodes=max_episodes, 
                                          no_ER=no_er, no_TN=no_tn)

    


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    modes = ['--ER', '--TN'] 
    no_er, no_tn = False, False
    
    if args is not None:
        if len(args) == 1 and args[0] == '--ER':
            no_er = True
        elif len(args) == 1 and args[0] == '--TN':
            no_tn = True
        elif len(args) == 2 and args[0] in modes and args[1] in modes:
            no_er = True
            no_tn = True
        else:
            print("Wrong argument(s)! You can only add two arguemnt: --ER or(and) --TN.")
            return 
        
    s = time.time()
    if args is None:
        # Chacking various NN architecture
        nn_experiment()

        # Checking various learning rate values
        # lr_experiment()

        # Checking various  exploration methods
        # gamma_experiment()
    else:
        ablation_study(no_er, no_tn)


    print("Program finished. Total time: {} seconds.".format(round(time.time() - s, 2)))


if __name__ == '__main__':
    #main()
    dqn_vs_ddqn()
