#  Here we can just play one of the modefrom
from Agent import DQNAgent
import time

def main():
    max_episodes = 10000
    npl = [32,32]
    solved=True
    
    learning_rate = 0.001
    gamma = 0.95, 

    policy = 'egreedy'
    epsilon = 0.5
    state_size = 4  
    action_size = 2  
    batch_size = 32 

    s = time.time()
    
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
    
    print("Program finished. Total time: {} seconds.".format(round(time.time()-s,2)))

if __name__ == '__main__':
    main()