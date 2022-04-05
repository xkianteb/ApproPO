import numpy as np
import time
import random
from gym import spaces
import gym
from collections import defaultdict
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional
import torch.nn.functional as F
from torch.distributions import Categorical,Normal

from collections import namedtuple
import itertools

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class RL_Oracle:
    def __init__(self, env=None, theta=None, net=None, args=None):
        super(RL_Oracle, self).__init__()
        self.env = env
        self.gamma = args.gamma
        self.lr = args.rl_lr
        self.saved_actions = []
        self.rewards = []
        self.entropies = []
        self.theta = theta
        self.eps = np.finfo(np.float32).eps.item()
        self.device = args.device

        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)#
        #self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr, eps=1e-5, alpha=0.99)
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        #self.diversity = args.diversity

    def reset(self, normalize_theta=True):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.entropies[:]

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action_scores, state_value = self.net(state)
        m = Categorical(logits=-action_scores)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        self.entropies.append(m.entropy())
        return action.item()

    def finish_episode(self, normalize_theta=None):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        #returns = torch.tensor(returns)
        #returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            R = torch.tensor([R]).to(self.device)
            value_losses.append(F.smooth_l1_loss(value, R.reshape(-1,1)))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).mean()\
             + (self.value_coef * torch.stack(value_losses).mean())\
             - (self.entropy_coef * torch.stack(self.entropies).mean())
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        self.reset(normalize_theta=normalize_theta)

    def learn_policy(self, n_traj=50, n_iter=300, update=True, normalize_theta=True, cost=False):
        self.reset(normalize_theta=normalize_theta)
        sum_measurements = np.zeros(np.shape(self.theta))
        episode_stats = defaultdict(list)

        for _ in range(n_traj): # ramdom in number of trajectory
            obs = self.env.reset()
            done = False
            traj_measurements = np.zeros(self.theta.size)

            for i in range(n_iter):
                action = self.select_action(obs)
                obs, env_reward, done, info = self.env.step(action)

                constraint = info['constraint']
                goal = info['goal']
                # print(f"constraint: {constraint} , goal :{goal}")
                measurements = np.append(np.array([-float(constraint), env_reward]), (obs==1)*1.0)

                traj_measurements = traj_measurements + measurements
                reward = np.dot(self.theta, measurements) # theta is lambda
                if cost:
                    reward = -reward
                self.rewards.append(reward)

                if done:
                    break

            episode_stats['traj_len'].append(i)
            if update:
                self.finish_episode(normalize_theta=normalize_theta)

            traj_measurements[2:] = traj_measurements[2:] / (i+1)
            sum_measurements = sum_measurements+traj_measurements

            episode_stats['goals'].append(goal * 1)

        #print(f'{obs}')
        #print('-----')
        avg_measurements = sum_measurements / n_traj  # long term measurement
        return (avg_measurements, episode_stats)
