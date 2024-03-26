# RL-Leiden-S24: Reinforcement Learning Assignment 2
## Goal: Training Deep Q-learning Network on Cart Pole environment

## File structure:
*Agent.py* contains the BaseAgent class. Executes selected action and returns results.

*DQN_learner.py* contains the DQNLearningAgent class and the dqn_learning function.
                  The class includes the following methods:
                    - Select action based on policy
                    - Update q-values
                    - Evaluate current strategy
                  The function runs the learner using DQNLearningAgent & BaseAgent classes and the Cart Pole environment class.

*Experiment.py* contains the functions for running varying learners for varying parameters, collecting the results.

*Helper.py* contains useful functions for use throughout the other files.

*Neural_network.py* contains the DeepNeuralNetwork class. Runs the fully connected convolutional network.
