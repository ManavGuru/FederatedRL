
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
no_of_clients = 1

class Gradient_calc():
    def __init__(self, agent, env, config: Config, record=False):

    def set_no_of_clients(no_of_c):
        global no_of_clients
        no_of_clients = no_of_c
        print("no_of_clients = ",no_of_clients)

    sum_gradients_actor = {}
    sum_gradients_critic = {}
    def reset_avg_gradients(agent):
        for name, param in agent.actor.to(device).named_parameters():
            sum_gradients_actor[name] = torch.zeros(size=param.size()).to(device)
        for name, param in agent.critic.to(device).named_parameters():
            sum_gradients_critic[name] = torch.zeros(size=param.size()).to(device)
        

    sum_weights_actor = {}
    sum_weights_critic = {}
    def reset_avg_weights(agent):
        for name, param in agent.actor.to(device).named_parameters():
            sum_weights_actor[name] = torch.zeros(size=param.size()).to(device)
        for name, param in agent.critic.to(device).named_parameters():
            sum_weights_critic[name] = torch.zeros(size=param.size()).to(device)

    def calc_avg_gradients(client):
        for i in range(len(client)):
            gradients_actor = {}
            for name, param in client[i].agent.actor.to(device).named_parameters():
                gradients_actor[name] = param.grad
                sum_gradients_actor[name] += (gradients_actor[name] / no_of_clients)
            gradients_critic = {}
            for name, param in client[i].agent.critic.to(device).named_parameters():
                gradients_critic[name] = param.grad
                sum_gradients_critic[name] += (gradients_critic[name] / no_of_clients)


    def cal_avg_weights(client):
        for i in range(len(client)):
            weights_actor = {}
            for name, param in client[i].agent.actor.to(device).named_parameters():
                weights_actor[name] = param
                sum_weights_actor[name] += (weights_actor[name] / no_of_clients)
            weights_critic = {}
            for name, param in client[i].agent.critic.to(device).named_parameters():
                weights_critic[name] = param
                sum_weights_critic[name] += (weights_critic[name] / no_of_clients)

    def cal_best_gradients(client):
        idx = np.argmax([np.mean(client[i].agent.scores_window) for i in range(len(client))])
        #print([np.mean(client[i].scores_window) for i in range(len(client))])
        #print("idx = ",idx)
        for name, param in client[idx].agent.actor.to(device).named_parameters():
            sum_gradients_actor[name] = param.grad
        for name, param in client[idx].agent.critic.to(device).named_parameters():
            sum_gradients_critic[name] = param.grad

    def set_new_gradients(agent):
        for name, param in agent.actor.to(device).named_parameters():
            param.grad = sum_gradients_actor[name].clone()
            param.grad.requires_grad=True
        agent.actor_optimizer.step()
        agent.actor_optimizer.zero_grad()
        for name, param in agent.critic.to(device).named_parameters():
            param.grad = sum_gradients_critic[name].clone()
            param.grad.requires_grad=True
        agent.critic_optimizer.step()
        agent.critic_optimizer.zero_grad()

    def set_new_weights(agent):
        for name, param in agent.actor.to(device).named_parameters():
            param = sum_weights_actor[name].clone()
        agent.actor_optimizer.step()
        agent.actor_optimizer.zero_grad()
        for name, param in agent.critic.to(device).named_parameters():
            param = sum_weights_critic[name].clone()
        agent.critic_optimizer.step()
        agent.critic_optimizer.zero_grad()

    def get_gradients(agent):
        gradients = {}
        for name, param in agent.local_model.to(device).named_parameters():
            gradients[name] = param.grad