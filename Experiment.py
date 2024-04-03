from Agent import DQNAgent, dqn_learner
import time
import numpy as np
import os
import sys
import pickle
import pandas as pd

import matplotlib.pyplot as plt


def make_plot(values, labels, max_episodes, title):
    x = np.arange(1, max_episodes +1)

    plt.figure(figsize=(10, 6))
    if isinstance(labels, list):
        for y,l in zip(values, labels):
            plt.plot(x, y, label=l)
    elif isinstance(labels, str):
        plt.plot(x, y, label=l)
    else:
        print("Error!")    

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xlim(0, max_episodes+1)
    plt.xticks(range(0,  max_episodes+int(0.1*max_episodes), int(0.1*max_episodes)))
    plt.title(title, fontsize=14)

    plt.legend(prop={'size': 12})
    plt.grid(True)
    plt.savefig("Pics/" + title +".png")
    # plt.show()


def save_run(scores, evals, general_title, test_title):
    path = "Data/"
    file_name_s = general_title + '_' + test_title + "_score.csv"
    file_name_e = general_title + '_' + test_title + "_evals.csv"

    if not os.path.exists(path):
        os.makedirs(path)
    try:
        # pickle to store lists instead of strings
        with open(path + file_name_s, "wb") as myfile:
            pd.DataFrame({'scores':scores}).to_csv(myfile)

        with open(path + file_name_e, "wb") as myfile:
            pd.DataFrame({'Evals':evals}).to_csv(myfile) 
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
    max_episodes = 150

    # test_title = 'dqn'
    # scores, evals = dqn_learner(ddqn=False,max_episodes=max_episodes)
    # save_run(scores, evals, general_title, test_title)

    test_title = 'ddqn'
    scores, evals = dqn_learner(ddqn=True, max_episodes=max_episodes)
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
    print("Starting tunning learning rate...")
    general_title = 'lr_experiment'
    max_episodes = 100
    learning_rates = [0.001]

    title = "Tune learning rate"
    scorces_list = []
    labels = ['lr='+str(x) for x in learning_rates]
    for lr in learning_rates:
        test_title = 'lr_'+str(lr)
        scores, evals = dqn_learner(learning_rate=lr, max_episodes=max_episodes)
        scorces_list.append(scores)
        save_run(scores, evals, general_title, test_title)

    make_plot(scorces_list, labels, max_episodes, title)


def gamma_experiment():
    general_title = 'nn_experiment'
    max_episodes = 150
    policies = ['egreedy', 'softmax']
    for policy in policies:
        test_title = str(policy)
        scores, evals = dqn_learner(policy=policy,max_episodes=max_episodes)
        save_run(scores, evals, general_title, test_title)


def ablation_study(no_er, no_tn):
    print("Starting ablation study...")
    max_episodes = 150
    npl = [24,24]
    
    learning_rate = 0.001 
    gamma = 0.9          

    policy = 'egreedy'
    epsilon = 1.0
    tau = 0.1
    batch_size = 24

    file_name_DQN_results = ".\Data\DQN_150.csv"
    scores_dqn = pd.read_csv(file_name_DQN_results, names=["scores"])["scores"].to_list()
    print(scores_dqn, len(scores_dqn))
    
    file_gen_title = "ablation_study"
    if no_er == True and no_tn == True:
        label1, label2 = "DQN", "DQN−ER−TN"
        title = label1 + " vs " + label2
        comp_name = "DQN−EP−TN"
    elif no_er == True and no_tn == False:
        label1, label2 = "DQN", "DQN−ER"
        title = label1 + " vs " + label2
        comp_name = "DQN−TN"
    else:
        label1, label2 = "DQN", "DQN−TN"
        title = label1 + " vs " + label2
        comp_name = "DQN−ER"
    
    scores_comp, evals_comp = dqn_learner(learning_rate=learning_rate,
                                          gamma=gamma,
                                          batch_size=batch_size,
                                          epsilon=epsilon,
                                          NPL=npl,
                                          tau=tau,
                                          policy=policy,
                                          max_episodes=max_episodes, 
                                          no_ER=no_er, no_TN=no_tn)
    save_run(scores_comp, evals_comp, file_gen_title, comp_name)


    make_plot([scores_dqn, scores_comp], [label1, label2], max_episodes, title)

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
        # nn_experiment()

        # Checking various learning rate values
        lr_experiment()

        # Checking various  exploration methods
        # gamma_experiment()
    else:
        ablation_study(no_er, no_tn)


    print("Program finished. Total time: {} seconds.".format(round(time.time() - s, 2)))


if __name__ == '__main__':
    main()
    # dqn_vs_ddqn()
