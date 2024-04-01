from Agent import DQNAgent
import time 
import numpy as np


import matplotlib.pyplot as plt

def plot_perf(loss_avg, stepst):
    pass

def main():
    learning_rate = 0.0001
    gamma = 0.98, 

    policy = 'egreedy'
    epsilon = 1.0
    state_size = 4  
    action_size = 2  
    batch_size = 32 
    # npl = [24, 24]
    npl = [128,128,128]

    
    print("Starting running...")
    s = time.time()
    agent = DQNAgent(state_size=state_size,
                     action_size=action_size,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     policy=policy,
                     batch_size=batch_size,
                     epsilon=epsilon,
                     npl=npl)
    
    loss_avg, steps = agent.run()
    plot_perf(loss_avg, steps)
    
    print("Program finished. Total time: {} seconds.".format(round(time.time()-s, 2)))

if __name__ == '__main__':
    main()
  