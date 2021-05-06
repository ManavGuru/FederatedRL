import os

import numpy as np
from gym import wrappers
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder, time_seq
import torch


class Trainer:
    def __init__(self, agent, env, config: Config, record=False):

        self.agent = agent
        self.config = config
        #self.outputdir = get_output_folder(self.config.output, self.config.env)

        if record:
            os.makedirs('video', exist_ok=True)
            filepath = self.outputdir + '/video/' + config.env + '-' + time_seq()
            env = wrappers.Monitor(env, filepath,
                                   video_callable=lambda episode_id: episode_id % self.config.record_ep_interval == 0)

        self.env = env
        self.env.seed(config.seed)

        self.agent.is_training = True

        #self.agent.save_config(self.outputdir)
        #self.board_logger = TensorBoardLogger(self.outputdir)

    def train(self, episode_per_fed, fed_round,i, pre_episodes=0, pre_total_step=0):
        total_step = pre_total_step

        #all_rewards = []
        #for ep in range(pre_episodes + 1, self.config.episodes + 1):

        for ep in range(1, episode_per_fed + 1):
            s0 = self.env.reset()
            self.agent.reset()

            done = False
            step = 0
            actor_loss, critics_loss, reward = 0, 0, 0

            # decay noise
            self.agent.decay_epsilon()

            while not done:
                action = self.agent.get_action(s0)

                s1, r1, done, info = self.env.step(action)
                self.agent.buffer.add(s0, action, r1, done, s1)
                s0 = s1

                if self.agent.buffer.size() > self.config.batch_size:
                    loss_a, loss_c = self.agent.learning()
                    actor_loss += loss_a
                    critics_loss += loss_c

                reward += r1
                step += 1
                total_step += 1

                if step + 1 > self.config.max_steps:
                    break

            self.agent.all_rewards.append(reward)
            self.agent.scores_window.append(reward)
            #self.agent.avg_reward = float(np.mean(all_rewards[-100:]))
            #self.board_logger.scalar_summary('Reward per episode', ep, all_rewards[-1])
            #self.board_logger.scalar_summary('Best 100-episodes average reward', ep, avg_reward)

            print('\r fed_round: %d, Client: %d, total step: %5d, episodes %3d, episode_step: %5d, episode_reward: %5f' % (
                fed_round, i+1, total_step, ep, step, reward), end="")
            #print(' fed_round: %d, Client: %d, total step: %5d, episodes %3d, episode_step: %5d, episode_reward: %5f' % (
            #    fed_round, i+1, total_step, ep, step, reward))

            if (episode_per_fed*fed_round + ep) % 100 == 0:
                print('\r fed_round: %d, Client: %d, total step: %5d, episodes %3d, episode_step: %5d, episode_reward: %5f' % (
                fed_round, i+1, total_step, ep, step, reward))

            # check point
            #if self.config.checkpoint and ep % self.config.checkpoint_interval == 0:
            #    self.agent.save_checkpoint(ep, total_step, self.outputdir)
        #print("  buffer size = ",self.agent.buffer.size())

        # save model at last
        #self.agent.save_model(self.outputdir)









