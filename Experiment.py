from Agent import DQNAgent
import time 
import numpy as np


import matplotlib.pyplot as plt

def plot_perf(loss_avg, stepst):
    pass

def main():
    lr = 0.0001
    gamma = 0.95
    policy = 'egreedy'
    epsilon = 0.1
    train_max = 128 # batch-size
    
    print("Starting running...")
    s = time.time()
    agent = DQNAgent(learning_rate=lr,
                     gamma=gamma,
                     policy=policy,
                     train_max=train_max,
                     epsilon=epsilon)
    
    loss_avg, steps = agent.run()
    plot_perf(loss_avg, steps)
    
    print("Program finished. Total time: {} seconds.".format(time.time()-s))

if __name__ == '__main__':
    s = time.time()
    main()
    print("\n Program finished it took {} seconds to run.".format(time.time()-s))