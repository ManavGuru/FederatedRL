The following project contains code for Federated Reinforcement Learning along with code for Concurrent RL.

=====================================================================================
DQN - Basic Federated Averaging code
global_buffer - DQN based code with concurrent RL implementation
DDPG2 - DDPG code with federation
DDPG2_concurrent - DDPG code with concurrent RL
new_results, Result_data - contains results and episode reward csv files.

=====================================================================================

Run python notebook file in the respective folders to run the code.
Requires - Python 3.6 and Pytorch.


======================================================================================

The federation parameters can be changed in the main notebook file - 
simple_avg - gradient averaging
best_score - for best score method
weight_avg - for weight averaging


The other parameters like federation rounds, episode per federation, save file name can also be changed in the main notebook file