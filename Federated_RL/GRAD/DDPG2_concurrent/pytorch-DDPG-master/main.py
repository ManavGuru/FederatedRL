import argparse
import os
import gym
from gym import wrappers
from config import Config
from core.normalized_env import NormalizedEnv
from core.util import time_seq, load_obj
from ddpg import DDPG
from tester import Tester
from trainer import Trainer
import matplotlib.pyplot as plt
import pandas as pd
#from grad_utils import reset_avg_gradients, reset_avg_weights, calc_avg_gradients, cal_avg_weights, cal_best_gradients, set_new_gradients, set_new_weights, set_no_of_clients
from grad_utils import gradient_utils
import torch

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train', action='store_true', help='train model')
parser.add_argument('--test', dest='test', action='store_true', help='test model')
parser.add_argument('--env', default='LunarLanderContinuous-v2', type=str, help='gym environment')
parser.add_argument('--gamma', default=0.99, type=float, help='discount')
parser.add_argument('--episodes', default=3, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epsilon', default=1.0, type=float, help='noise epsilon')
parser.add_argument('--eps_decay', default=0.001, type=float, help='epsilon decay')
parser.add_argument('--max_buff', default=10000000, type=int, help='replay buff size')
parser.add_argument('--output', default='out', type=str, help='result output dir')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='use cuda')
parser.add_argument('--model_path', type=str, help='if test mode, import the model')
parser.add_argument('--load_config', type=str, help='load the config from obj file')

step_group = parser.add_argument_group('step')
step_group.add_argument('--customize_step', dest='customize_step', action='store_true', help='customize max step per episode')
step_group.add_argument('--max_steps', default=1000, type=int, help='max steps per episode')

record_group = parser.add_argument_group('record')
record_group.add_argument('--record', dest='record', action='store_true', help='record the video')
record_group.add_argument('--record_ep_interval', default=20, type=int, help='record episodes interval')

checkpoint_group = parser.add_argument_group('checkpoint')
checkpoint_group.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='use model checkpoint')
checkpoint_group.add_argument('--checkpoint_interval', default=500, type=int, help='checkpoint interval')

retrain_group = parser.add_argument_group('retrain')
retrain_group.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
retrain_group.add_argument('--retrain_model', type=str, help='retrain model path')

args = parser.parse_args()
config = Config()
config.env = args.env
#config.env = 'LunarLanderContinuous-v2'
# Pendulum-v0 # Walker2d-v2 # HalfCheetah-v1
config.gamma = args.gamma
config.episodes = args.episodes
config.max_steps = args.max_steps
config.batch_size = args.batch_size
config.epsilon = args.epsilon
config.eps_decay = args.eps_decay
config.max_buff = args.max_buff
config.output = args.output
config.use_cuda = args.cuda
config.checkpoint = args.checkpoint
config.checkpoint_interval = args.checkpoint_interval

config.learning_rate = 1e-3
config.learning_rate_actor = 1e-4
config.epsilon_min = 0.001
config.epsilon = 1.0
config.tau = 0.001

# env = gym.make() is limited by TimeLimit, there is a default max step.
# If you want to control the max step every episode, do env = gym.make(config.env).env
env = None
if args.customize_step:
    env = gym.make(config.env).env
else:
    env = gym.make(config.env)

#env = NormalizedEnv(env)
config.action_dim = int(env.action_space.shape[0])
config.action_lim = float(env.action_space.high[0])
config.state_dim = int(env.observation_space.shape[0])

print("env = ",env)
print("para 1",config.action_dim)
print("para 2",config.action_lim)
print("para 3",config.state_dim)

if args.load_config is not None:
        config = load_obj(args.load_config)

#------------------Federation Parameters
no_of_clients = 5
clients = []
fed_rounds = 450
episode_per_fed = 4
seed_val = 19
grad_update_method = 'global_buff'
window_size = 100

#-----------------initialze gradient code
grad = gradient_utils(no_of_clients)

#------------------Initialize all agents
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i in range(no_of_clients):
    agent = DDPG(seed_val,config)
    trainer = Trainer(agent, env, config,
                      record=args.record)
    clients.append(trainer)

for i in range(no_of_clients):
    print(clients[i])


print("***********************************************************************")

#------------------Run the federation
for f in range(fed_rounds):
    if grad_update_method == 'simple_avg' or grad_update_method == 'best_score':
        grad.reset_avg_gradients(clients[0].agent)
    elif grad_update_method == 'weight_avg':
        grad.reset_avg_weights(clients[0].agent)
    else:
        x = 10 #do nothing as of now

    for i in range(no_of_clients):
        clients[i].train(episode_per_fed,f,i)
    
    if grad_update_method == 'simple_avg': 
        grad.calc_avg_gradients(clients)
    elif grad_update_method == 'best_score':
        grad.cal_best_gradients(clients)
    elif grad_update_method == 'weight_avg':
        grad.cal_avg_weights(clients)
    else:
        x = 10 #do nothing as of now

    for i in range(no_of_clients):
        if grad_update_method == 'simple_avg' or grad_update_method == 'best_score':
            grad.set_new_gradients(clients[i].agent)
        elif grad_update_method == 'weight_avg':
            grad.set_new_weights(clients[i].agent)
        else:
            x = 10 #do nothing as of now


#if args.load_config is not None:
#        config = load_obj(args.load_config)

#agent = DDPG(config)
#trainer = Trainer(agent, env, config,record=args.record)
#trainer.train()
#print("rewards = ",agent.all_rewards)

def moving_average(vec,window_size):    

    numbers_series = pd.Series(vec)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    averaged_v = moving_averages_list[window_size - 1:]
    return(averaged_v)

def plot_graphs(values1,values2,xstring,ystring,title): 
  
    y1 = values1
    y2 = values2
    x1 = [i for i in range(0,len(values1))]
    x2 = [i for i in range(0,len(values2))]
    plt.plot(x1,y1)
    plt.plot(x2,y2,linewidth = 4)
    plt.xlabel(xstring) # naming the x axis      
    plt.ylabel(ystring) # naming the y axis
    plt.title(title) 

    plt.show()


averaged_scr = []
for i in range(no_of_clients):   
    averaged_ = moving_average(clients[i].agent.all_rewards,window_size)
    #averaged_scr.append(averaged_)
    #plot_graphs(clients[i].scores,averaged_,'epsiodes','rewards',' rewards vs episodes')
    #print(averaged_[0:20])
    averaged_scr.append(averaged_)
avg_score = averaged_

df2 = pd.DataFrame(averaged_scr)
df2 = df2.transpose()
df2.to_csv('ddpg_Concurrent_seed19.csv')
print(df2)

'''
if args.train:
    trainer = Trainer(agent, env, config,
                      record=args.record)
    trainer.train()

elif args.retrain:
    if args.retrain_model is None:
        print('please add the retrain model path:', '--retrain_model xxxx')
        exit(0)

    ep, step = agent.load_checkpoint(args.retrain_model)
    trainer = Trainer(agent, env, config,
                      record=args.record)
    trainer.train(ep, step)


elif args.test:
    if args.model_path is None:
        print('please add the model path:', '--model_path xxxx')
        exit(0)

    # record
    if args.record:
        os.makedirs('video', exist_ok=True)
        filepath = 'video/' + args.env + '-' + time_seq()
        env = wrappers.Monitor(env, filepath, video_callable=lambda episode_id: episode_id % 25 == 0)

    tester = Tester(agent, env,
                    model_path=args.model_path)
    tester.test()

else:
    print('choose train or test:', '--train or --test')
'''