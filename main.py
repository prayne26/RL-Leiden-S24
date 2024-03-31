#  Here we can just play one of the modefrom
from Agent import DQNAgent
import time

def main():
    lr = 0.0001
    gamma = 0.95
    policy = 'egreedy'
    epsilon = 0.1
    batch_size = 32 # batch-size

    s = time.time()
    agent = DQNAgent(learning_rate=lr,
                     gamma=gamma,
                     policy=policy,
                     batch_size=batch_size,
                     epsilon=epsilon)
    
    agent.run()

    print("Program finished. Total time: {} seconds.".format(time.time()-s))

if __name__ == '__main__':
    main()