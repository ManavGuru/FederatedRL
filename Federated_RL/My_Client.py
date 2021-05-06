import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
from My_DDQN import DDQN
import gym 
import time
from IPython.display import display, clear_output
class FederatedClient:
	def __init__ (self, env_name,seed_val,no_of_episodes, client_name): 
		self.env_name = env_name
		self.client_name = client_name
		self.no_of_episodes = no_of_episodes
		self.reward_list = []
		self.agent, self.env = self.create_one_client(seed_val,self.env_name)
		# self.score = 0 #to be used to eval client for gradient sharing
	def create_one_client(self, i, env_name):
		env = gym.make(env_name)
		env.seed(i)
        # env = gym.make('LunarLander-v2')
		observation_space = env.observation_space.shape[0]
		agent = DDQN(observation_space, env.action_space.n)
		return agent,env

	def client_train(self,fed_round):
	    episodes = self.no_of_episodes
	    t1=time.time()
	    max_time=1000
	    for e in range(episodes):
	        episode_reward=0
	        state=self.env.reset()  #[0.5,1,5]  
	        state = np.reshape(state, [1, self.agent.nS])  #[[0.5,1,5]]  
	        t=0
	        while True:
	            t+=1
	            action=self.agent.act(state)        
	            state_next, reward, terminal,info = self.env.step(action)
	            episode_reward+=reward
	            state_next = np.reshape(state_next, [1, self.agent.nS])
	            self.agent.add_memory(state, action, reward, state_next, terminal)
	            state = state_next
	            if(len(self.agent.memory)>self.agent.minibatch_size):
	                self.agent.replay()
	            if terminal or (t>max_time):            
	                clear_output(wait=True)
	                display("Fed Round : " + str(fed_round) + " , Client ID : " + self.client_name+": "+"Episode: " + str(e) + ", exploration: " + str(self.agent.exploration_rate) + ", score: " + str(episode_reward))              
					#display('Time Elapsed '+str(time.time()-t1))
	                self.agent.target_model_update()
	                self.reward_list.append(episode_reward)
	                if(e>150):
	                    display('Mean is ' +str(np.mean(self.reward_list[len(self.reward_list)-100:])))
	                break 
	        if self.agent.exploration_rate>self.agent.exploration_min:
	            self.agent.exploration_rate*=self.agent.exploration_decay

    #HelperFN: Getting moving averages to smoothen plots
	def get_moving_average(self,N=100):
		cum_sum, moving_aves = [0], []
		for i, x in enumerate(self.reward_list, 1):
			cum_sum.append(cum_sum[i-1] + x)
			if i>=N:
				moving_ave = (cum_sum[i] - cum_sum[i-N])/N       
				moving_aves.append(moving_ave)
		return moving_aves 
############################################################################
## Evaluate client and update self.score() before sending gradients to federation controller. 
	# def agent_evaluate(self,agent):
	# 	env = gym.make('CartPole-v0')          
	# 	state=env.reset()  #[0.5,1,5]  
	# 	state = np.reshape(state, [1, agent.nS])  #[[0.5,1,5]]  
	# 	t=0
	# 	max_episode = 10
	# 	max_t = 1000
	# 	avg_episode_reward = 0
	# 	for e in range(max_episode):
	# 	    episode_reward=0
	# 	    while (t < max_t):
	# 	        t+=1
	# 	        action=agent.act(state)        
	# 	        state_next, reward, terminal,info = env.step(action)
	# 	        episode_reward+=reward
	# 	        state_next = np.reshape(state_next, [1, agent.nS])
	# 	        state = state_next
	# 	        if terminal:
	# 	            avg_episode_reward += episode_reward
	# 	            break 
	# 	return avg_episode_reward/max_episode


