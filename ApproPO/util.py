from collections import namedtuple
import uuid
import copy
import torch
import numpy as np

from scipy.stats import entropy
from numpy import linalg as LA
from ApproPO.envs.gym_frozenmarsrover.envs.maps import MAPS

# Name Tuple for storing items in cache
CacheItem = namedtuple('CacheItem', ['exp_rtn', 'exp_stats',\
                                     'shadow_rtn', 'shadow_stats',\
                                     'policy', 'uuid'])

def init_cache(rl_oracle_generator=None, args=None):
    """
    Method for initializing the cache with random oracles
    """
    cache_size = args.cache_size
    cache = []
    samples = 0
    trajs = 0
    for _ in range(cache_size):
        rl_oracle = rl_oracle_generator()
        with torch.no_grad():
            [exp_rtn, exp_stats] = rl_oracle.learn_policy(n_traj=args.check_traj, update=False, cost=True)
        cache.append(CacheItem(copy.deepcopy(exp_rtn), copy.deepcopy(exp_stats),\
                               copy.deepcopy(exp_rtn), copy.deepcopy(exp_stats),\
                               rl_oracle.net.state_dict(),\
                               uuid.uuid1()))
        samples += sum(exp_stats['traj_len'])
        trajs += len(exp_stats['traj_len'])
    return (samples, trajs, cache)

n_bins = 64
mask = np.triu(np.ones(n_bins).reshape(8,8)).flatten()
map = MAPS['8x8']
map = [list(x) for x in map]
map = [item for sublist in map for item in sublist]
map = np.where('H' == np.array(map))[0].tolist()
map.extend([63, 0])
mask[map] = 0
bins = np.ones(64)
bins = bins * mask
bins = bins / np.sum(bins)


#norm = lambda raw: np.array([float(i)/sum(raw) for i in raw])
#def norm(raw):
#    return np.array([float(i)/sum(raw) for i in raw])

#calc_entropy = lambda x: entropy(norm(x))
#def calc_entropy(x):
#    return entropy(norm(x))

#calc_dist_uni = lambda x: LA.norm(x - bins, ord=2)
def calc_dist_uni(x):
    return LA.norm(x - bins, ord=2)
