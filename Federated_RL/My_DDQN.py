#Desik's DDQN Code
#This is a  Deep Q network 
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
#In this code, the Q-Network takes in state as input and gives value as output
class DDQN:
	def __init__(self,nS,nA):
		self.nS=nS #Number of observation/States
		self.nA=nA	#Number of action
		self.exploration_rate=1	 
		self.exploration_min=0.01
		self.exploration_decay=0.995
		self.gamma=0.99
		self.learning_rate=1e-4
		self.verbose=0
		self.rewards_list= list()
		self.minibatch_size=64
		self.memory=deque(maxlen=50000) #Expreience replay size		
		self.model=self.create_model() #calls the create model
		self.target_model=self.create_model() #Target Network
        
        
	def create_model(self): #we do this to keep 2 networks
		model=Sequential() 
		model.add(Dense(256, input_shape=(self.nS,),activation="relu"))
#		model.add(Dense(64,activation="relu"))
# 		model.add(Dense(64,activation="relu"))
		model.add(Dense(self.nA,activation="linear"))
		model.compile(loss="mse",optimizer=Adam(lr=self.learning_rate))
		return model

	def add_memory(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done)) #adding to the exprience replay memory
	

	def target_model_update(self):
		self.target_model.set_weights(self.model.get_weights()) #set the weights of the target network using the weights of the original network 

	def act(self,state):
		if np.random.rand()<self.exploration_rate:
			return np.random.choice(self.nA)	#exploration
		q=self.model.predict(state) #Choosing the optimal action
		return np.argmax(q[0])

	def replay(self):
		#does the iteration using for loops				
		minibatch=random.sample(self.memory,self.minibatch_size) #sample from expericence replay		
		state = np.array([sars[0] for sars in minibatch])
		action = np.array([sars[1] for sars in minibatch])
		reward = np.array([sars[2] for sars in minibatch])
		next_state = np.array([sars[3] for sars in minibatch])
		done = np.array([sars[4] for sars in minibatch])
		state = np.squeeze(state)
		next_state = np.squeeze(next_state)
		target = reward + self.gamma*(np.amax(self.target_model.predict_on_batch(next_state), axis=1))*(1-done)
		update = self.model.predict_on_batch(state)
		index = np.array([i for i in range(self.minibatch_size)])
		update[[index],[action]] = target
		self.model.fit(state, update, epochs=1, verbose=0)	

	def get_model_weights(self):
		return self.model.get_weights()

	def set_weight(self,weights):
		self.model.set_weights(weights)
        
	def set_reward(self, reward): 
		self.reward_list.append(reward)

	def get_reward(self): 
		return self.reward_list
