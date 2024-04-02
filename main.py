#  Here we can just play one of the modefrom
from Agent import DQNAgent, dqn_learner
import sys
import time

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
        

    max_episodes = 10000
    npl = [32,32]
    
    learning_rate = 0.001 # Here should be tuned value
    gamma = 0.95          # Here should be tuned value

    policy = 'egreedy'
    epsilon = 0.8
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
                        NPL=npl,
                        tau=0.1)
    
    scores, evals = dqn_learner(NPL=npl, max_episodes=max_episodes, no_ER=no_er, no_TN=no_tn)
    
    print("Program finished. Total time: {} seconds.".format(round(time.time()-s,2)))

if __name__ == '__main__':
    main()