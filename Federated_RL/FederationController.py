from My_DDQN import DDQN
import random
import numpy as np
import gym 
from My_Client import FederatedClient
class FederationController:
	def __init__(self, env_name, no_of_clients, seed_val, no_of_episodes): 
		self.env_name = env_name
		self.no_of_clients = no_of_clients
		self.no_of_episodes = no_of_episodes
		self.my_clients = self.create_clients(seed_val,no_of_clients)

	def create_clients(self,seed_val,no_of_clients):
	    clients = []
	    for i in range (self.no_of_clients): 
	        c = FederatedClient(env_name=self.env_name,seed_val = seed_val,
	        	no_of_episodes = self.no_of_episodes, client_name= str(i+1))
	        clients.append(c)
	    return clients

	def agent_evaluate(self,agent):
		env = gym.make(self.env_name)          
		state=env.reset()  #[0.5,1,5]  
		state = np.reshape(state, [1, agent.nS])  #[[0.5,1,5]]  
		t=0
		max_episode = 10
		max_t = 1000
		avg_episode_reward = 0
		for e in range(max_episode):
		    episode_reward=0
		    while (t < max_t):
		        t+=1
		        action=agent.act(state)        
		        state_next, reward, terminal,info = env.step(action)
		        episode_reward+=reward
		        state_next = np.reshape(state_next, [1, agent.nS])
		        state = state_next
		        if terminal:
		            avg_episode_reward += episode_reward
		            break 
		return avg_episode_reward/max_episode

	def simple_average(self): 
		n_layers = len(self.my_clients[0].agent.get_model_weights())
		avg_weight = list()
		for layer in range(n_layers): 
			layer_weights = ([c.agent.get_model_weights()[layer]for c in self.my_clients])
			avg_layer_weights = sum(layer_weights)/len(self.my_clients)
			avg_weight.append(avg_layer_weights)
		for c in self.my_clients: 
			c.agent.set_weight(avg_weight)

	def weighted_averaging(self): 
	    weights = []
	    for client in self.my_clients: 
	        weights.append(self.agent_evaluate(client.agent))
	    for i in range(len(weights)):
	      weights[i] = weights[i]/sum(weights)
	    n_layers = len(self.my_clients[0].agent.get_model_weights())
	    avg_weight = list()
	    for layer in range(n_layers): 
	        layer_weights = ([c.agent.get_model_weights()[layer]*weights[self.my_clients.index(c)] for c in self.my_clients])
	        avg_layer_weights = sum(layer_weights)
	        avg_weight.append(avg_layer_weights)
	    for c in self.my_clients: 
	        c.agent.set_weight(avg_weight)


	def max_weight(self):
	    weight_idx = np.argmax([self.agent_evaluate(client.agent) for client in self.my_clients])
	    max_weight = self.my_clients[weight_idx].agent.get_model_weights()
	    for c in self.my_clients: 
	        c.agent.set_weight(max_weight)

	def federated_training(self,federation_frequency, averaging_method = 'simple'): 
		rewards = []
		for i in range(len(self.my_clients)):
			rewards.append([])
		#for i in range (self.no_of_episodes):
		for i in range (federation_frequency):
			count = 1
			for client in self.my_clients:
				client.client_train(i)
				print("round = ",i,"  client = ",count)
				count += 1
			if(averaging_method == 'max'): 
				self.max_weight()
			elif(averaging_method == 'weighted'): 
				self.weighted_averaging()
			else:
				self.simple_average()
			count = 0
		return rewards