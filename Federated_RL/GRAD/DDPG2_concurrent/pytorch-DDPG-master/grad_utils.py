
import torch
import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#no_of_clients = 1

class gradient_utils():
    def __init__(self,no_of_clients):
        super(gradient_utils, self).__init__()

        self.sum_gradients_actor = {}
        self.sum_gradients_critic = {}
        self.sum_weights_actor = {}
        self.sum_weights_critic = {}
        self.no_of_clients = no_of_clients
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # def set_no_of_clients(no_of_c):
    #     global no_of_clients
    #     no_of_clients = no_of_c
    #     print("no_of_clients = ",no_of_clients)

    def reset_avg_gradients(self,agent):
        for name, param in agent.actor.to(self.device).named_parameters():
            self.sum_gradients_actor[name] = torch.zeros(size=param.size()).to(self.device)
        for name, param in agent.critic.to(self.device).named_parameters():
            self.sum_gradients_critic[name] = torch.zeros(size=param.size()).to(self.device)        


    def reset_avg_weights(self,agent):
        for name, param in agent.actor.to(self.device).named_parameters():
            self.sum_weights_actor[name] = torch.zeros(size=param.size()).to(self.device)
        for name, param in agent.critic.to(self.device).named_parameters():
            self.sum_weights_critic[name] = torch.zeros(size=param.size()).to(self.device)

    def calc_avg_gradients(self,client):
        for i in range(len(client)):
            gradients_actor = {}
            for name, param in client[i].agent.actor.to(self.device).named_parameters():
                gradients_actor[name] = param.grad
                self.sum_gradients_actor[name] += (gradients_actor[name] / self.no_of_clients)
            gradients_critic = {}
            for name, param in client[i].agent.critic.to(self.device).named_parameters():
                gradients_critic[name] = param.grad
                self.sum_gradients_critic[name] += (gradients_critic[name] / self.no_of_clients)


    def cal_avg_weights(self,client):
        for i in range(len(client)):
            weights_actor = {}
            for name, param in client[i].agent.actor.to(self.device).named_parameters():
                weights_actor[name] = param
                self.sum_weights_actor[name] += (weights_actor[name] / self.no_of_clients)
            weights_critic = {}
            for name, param in client[i].agent.critic.to(self.device).named_parameters():
                weights_critic[name] = param
                self.sum_weights_critic[name] += (weights_critic[name] / self.no_of_clients)

    def cal_best_gradients(self,client):
        idx = np.argmax([np.mean(client[i].agent.scores_window) for i in range(len(client))])
        #print([np.mean(client[i].scores_window) for i in range(len(client))])
        #print("idx = ",idx)
        for name, param in client[idx].agent.actor.to(self.device).named_parameters():
            self.sum_gradients_actor[name] = param.grad
        for name, param in client[idx].agent.critic.to(self.device).named_parameters():
            self.sum_gradients_critic[name] = param.grad

    def set_new_gradients(self,agent):
        for name, param in agent.actor.to(self.device).named_parameters():
            param.grad = self.sum_gradients_actor[name].clone()
            param.grad.requires_grad=True
        agent.actor_optimizer.step()
        agent.actor_optimizer.zero_grad()
        for name, param in agent.critic.to(self.device).named_parameters():
            param.grad = self.sum_gradients_critic[name].clone()
            param.grad.requires_grad=True
        agent.critic_optimizer.step()
        agent.critic_optimizer.zero_grad()

    def set_new_weights(self,agent):
        for name, param in agent.actor.to(self.device).named_parameters():
            param = self.sum_weights_actor[name].clone()
        agent.actor_optimizer.step()
        agent.actor_optimizer.zero_grad()
        for name, param in agent.critic.to(self.device).named_parameters():
            param = self.sum_weights_critic[name].clone()
        agent.critic_optimizer.step()
        agent.critic_optimizer.zero_grad()

    def get_gradients(self,agent):
        gradients = {}
        for name, param in agent.local_model.to(self.device).named_parameters():
            gradients[name] = param.grad