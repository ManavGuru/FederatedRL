from Qnetwork import QNetwork
from Replay_buffer import ReplayBuffer


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import numpy as np
import random
from collections import namedtuple, deque
import pandas as pd
import matplotlib.pyplot as plt

#=================Agent to interact and learn from the environment====================#

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = int(1e5)  # replay buffer size
        self.batch_size = 64         # minibatch size
        self.gamma = 0.99            # discount factor
        self.tau = 1e-3              # for soft update of target parameters
        self.lr = 1e-3               # learning rate 
        self.update_every = 2        # how often to update the network
        self.eps_start=1.0           #starting epsilon value
        self.eps_end=0.01            #minimum epsilon value
        self.eps_decay=0.995 
        self.eps = self.eps_start
        self.max_steps=1000          #max number of steps in an episode
            
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.local_model = QNetwork(state_size, action_size, seed).to(self.device)
        self.target_model = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self,state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_model.eval()
        with torch.no_grad():
            action_values = self.local_model(state)
        self.local_model.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
 
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.local_model(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_model, self.target_model, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)