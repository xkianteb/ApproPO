import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import gym
from gym import spaces
import time
import argparse
import operator
import pandas as pd
import scipy
from numpy import linalg as LA
from datetime import datetime
from scipy.stats import entropy
import uuid
from collections import defaultdict

from ApproPO.util import init_cache, CacheItem
from ApproPO.envs.gym_frozenmarsrover.envs.maps import MAPS
from ApproPO.policy import MixturePolicy
from ApproPO.util import calc_dist_uni

def run(proj_oracle=None, rl_oracle_generator=None, args=None):
    policy = MixturePolicy()
    shadow = MixturePolicy()

    stats = defaultdict(int)
    [num_samples, num_trajs, cache] = init_cache(rl_oracle_generator=rl_oracle_generator, args=args)
    stats['num_samples'] += num_samples
    stats['num_trajs'] += num_trajs

    init = True
    value = float("inf")
    theta = proj_oracle.get_theta()
    new_cache_item = True
    counter= 0
    for episode in range(args.num_epochs):
        min_value = float("inf")
        min_exp_rtn = None
        min_params = None
        min_stats = None
        min_uuid = None

        if value < 0 or np.isclose(0,value, atol=args.atol_cache, rtol=args.rtol_cache) or init:
        #if value < 0 or init:
            reset= True
            # Find the policy with the smallest expected return for
            # our current theta value
            for item in cache:
                value = np.dot(theta, np.append(item.exp_rtn, args.mx_size))
                if value < min_value or init:
                    min_value = value
                    min_exp_rtn = item.exp_rtn
                    min_stats = item.exp_stats
                    min_shadow_exp_rtn = item.shadow_rtn
                    min_shadow_stats = item.shadow_stats
                    min_params = item.policy
                    min_uuid = item.uuid
                    init=False

            if min_value == float("inf"):
                min_params = rl_oracle.net.state_dict()
                min_exp_rtn = best_exp_rtn
                min_stats = best_exp_stats
                min_shadow_exp_rtn = shadow_exp_rtn
                min_shadow_stats = shadow_exp_stats
                min_uuid = cache_uuid

        # Set expected return if min_value less than 0
        #if min_value < 0 or np.isclose(0,min_value, atol=1e-1):
        if reset:
            best_exp_rtn = min_exp_rtn
            best_exp_stats = min_stats
            shadow_exp_rtn = min_shadow_exp_rtn
            shadow_exp_stats = min_shadow_stats
            cache_uuid = min_uuid

            print(f'Warm Start: {min_uuid}')
            rl_oracle = rl_oracle_generator()
            new_params = rl_oracle.net.state_dict()
            new_params.update(min_params)
            rl_oracle.net.load_state_dict(new_params)
            rl_oracle.theta = theta[:-1]
            #rl_oracle.reset()
            stats['cache_calls'] += 1

        # Run RL Oracle and compute expected return
        else:
            rl_oracle.theta = theta[:-1] # last element is artificial (makes the cone)
            #rl_oracle.reset()
            [best_exp_rtn, best_exp_stats] =\
                 rl_oracle.learn_policy(n_traj=args.rl_traj, n_iter=args.rl_iter, cost=True)

            with torch.no_grad():
                 [shadow_exp_rtn, shadow_exp_stats] =\
                      rl_oracle.learn_policy(n_traj=args.check_traj, n_iter=args.rl_iter, update=False, cost=True)

            stats['num_samples'] += sum(best_exp_stats['traj_len'])
            stats['num_trajs'] += len(best_exp_stats['traj_len'])
            stats['oracle_calls'] += 1
            stats['goals'] = np.average(best_exp_stats['goals'])
            new_cache_item = True

        # Update depends on current oracle result
        reset = False
        value = np.dot(theta, np.append(best_exp_rtn, args.mx_size))
        if value < 0 or np.isclose(0,value, atol=args.atol_cache, rtol=args.rtol_cache):
            counter = 0
            proj_oracle.update(best_exp_rtn.copy()) # Update OLO
            theta = proj_oracle.get_theta()
            print(f"New theta: {theta[:2]} -- value: {value}")

            if new_cache_item:
                cache_uuid = uuid.uuid1()
                cache.append(CacheItem(best_exp_rtn, best_exp_stats,\
                                       shadow_exp_rtn, shadow_exp_stats,\
                                       rl_oracle.net.state_dict(), cache_uuid))
                new_cache_item = False

            stats['episode'] = episode
            shadow.add_response(best_exp_rtn=best_exp_rtn, traj_len=0, stats=stats)
            policy.add_response(best_exp_rtn=best_exp_rtn, traj_len=0, stats=stats)

            dist_to_target = np.linalg.norm(policy.exp_rtn_of_avg_policy\
                                      - proj_oracle.proj(policy.exp_rtn_of_avg_policy))

            # Stats
            status = f'Epoch: [{episode}/{args.num_epochs}]\n'
            status += f'  exp_rtn_of_avg_policy: {policy.exp_rtn_of_avg_policy[:2]}\n'
            status += f'  best_exp_rtn: {best_exp_rtn[:2]}\n'
            status += f'  shadow_rtn_of_avg_policy: {shadow.exp_rtn_of_avg_policy[:2]}\n'
            status += f'  exp_rtn_goals: {policy.exp_rtn_goal}\n'
            status += f'  dist-to-target: {dist_to_target}\n'
            status += f'  dist_uni: {calc_dist_uni(policy.exp_rtn_of_avg_policy[2:])}\n'
            status += f'  trajectories: {stats["num_trajs"]}\n'
            status += f'  Samples: {stats["num_samples"]}'
        else:
            print(f'Old theta: {rl_oracle.theta}')
            counter+=1
            # Stats
            status = f'Epoch: [{episode}/{args.num_epochs}]\n'
            status += f'   NO UPDATE, <theta,u> is equal: {value}\n'
            status += f'   best_exp_rtn: {best_exp_rtn[:2]}\n'
            status += f'   trajectories: {stats["num_trajs"]}\n'
            status += f'   Samples: {stats["num_samples"]}'
            #rl_oracle.reset()

        print(status)
        print("------------------------------------------------")
        # if counter == 1000:
        #     print("NO IMPROVEMENT over 1000 epoch. STOP NOW!")
        #     break

    if args.print:
        now = datetime.now() # current date and time
        date = now.strftime("%Y%m%d%H%M%S")

        policy_df = pd.DataFrame(policy.stats,\
                 columns=np.hstack(['ep', 'traj', 'reward', 'prob_failure',\
                                    'obs', 'dist_uni', 'entropy', 'oracle_calls',\
                                    'cache_calls', 'traj_len', 'avg_traj_len', 'samples']))

        shadow_df = pd.DataFrame(shadow.stats,\
                 columns=np.hstack(['ep', 'traj', 'reward', 'prob_failure',\
                                    'obs', 'dist_uni', 'entropy', 'oracle_calls',\
                                    'cache_calls', 'traj_len', 'avg_traj_len', 'samples']))
        root_dir = '/home/anh/ApproPO'
        if args.diversity:
            policy_df.to_csv(f'{root_dir}/{args.output}/prob_failure_{args.prob_failure}_{args.init_variable}_threshold_{args.threshold}_diversity.csv', index=False)
            shadow_df.to_csv(f'{root_dir}/{args.output}/ours_best_{args.name}_{args.prob_failure}_{args.init_variable}_{args.threshold}_diversity.csv', index=False)
        else:
            policy_df.to_csv(f'{root_dir}/{args.output}/prob_failure_{args.prob_failure}_{args.init_variable}_threshold_{args.threshold}.csv', index=False)
            shadow_df.to_csv(f'{root_dir}/{args.output}/ours_best_{args.name}_{args.prob_failure}_{args.init_variable}_{args.threshold}.csv', index=False)

