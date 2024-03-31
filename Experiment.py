from Agent import DQNAgent
import time 
import numpy as np


import matplotlib.pyplot as plt

def plot_perf(loss_avg, stepst):
    pass

def main():
    lr = 0.001
    gamma = 0.99
    policy = 'egreedy'
    epsilon = 0.1
    batch_size = 32 # batch-size
    nlp = [24]
    nlp = [128,128,128]

    
    print("Starting running...")
    s = time.time()
    agent = DQNAgent(learning_rate=lr,
                     gamma=gamma,
                     policy=policy,
                     batch_size=batch_size,
                     epsilon=epsilon,
                     nlp=nlp)
    
    loss_avg, steps = agent.run()
    plot_perf(loss_avg, steps)
    
    print("Program finished. Total time: {} seconds.".format(round(time.time()-s, 2)))

if __name__ == '__main__':
    main()
  