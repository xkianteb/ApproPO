import numpy as np
from ApproPO.util import calc_dist_uni
from collections import deque

class MixturePolicy:
    def __init__(self):
        self.goals = []

        self.loss_vec = []
        self.stats = []

    def add_response(self, best_exp_rtn=None, traj_len=None, stats=None):
        self.loss_vec.append(best_exp_rtn)
        self.exp_rtn_of_avg_policy = np.average(np.stack(self.loss_vec, axis=0), axis=0)

        try:
            self.goals.append(stats['goals'])
            self.exp_rtn_goal = np.average(self.goals)
        except:
            raise Exception('Error: goal is not being passsed')

        self.stats.append(np.hstack([stats['episode'], stats['num_trajs'],\
              self.exp_rtn_of_avg_policy[1], self.exp_rtn_of_avg_policy[0],\
              {'obs': self.exp_rtn_of_avg_policy[2:].tolist()},\
              calc_dist_uni(self.exp_rtn_of_avg_policy[2:]),\
              0,\
              stats['oracle_calls'], stats['cache_calls'], {'lens': traj_len}, np.mean(traj_len),\
              stats['num_samples']]))

class RCPOPolicy:
    def __init__(self, shadow=False):
        self.goals = []
        self.cache_calls = 0
        self.oracle_calls = 0
        self.num_samples = 0
        self.num_trajs = 0
        self.stats = []

    def add_response(self, episode_stats=None, episode=None, rtn=None):
        self.num_samples += sum(episode_stats['traj_len'])
        self.num_trajs += len(episode_stats['traj_len'])
        self.oracle_calls += 1

        self.stats.append(np.hstack([episode, self.num_trajs, rtn[1], rtn[0],\
            {'obs': rtn[2:].tolist()}, calc_dist_uni(rtn[2:]), 0,\
            self.oracle_calls, self.cache_calls, {'lens': episode_stats['traj_len']},\
            np.mean(episode_stats['traj_len']), self.num_samples, episode_stats['_lambda'][0] ]))
