from datetime import datetime
import argparse
import random
import pandas as pd
import gym
import numpy as np
import torch
from collections import defaultdict

from ApproPO.nets import MLP
from ApproPO.envs import gym_frozenmarsrover
from ApproPO.rl_oracle_actor_critic import RL_Oracle as RL_Oracle_Actor_Critic
from ApproPO.policy import RCPOPolicy, MixturePolicy
from ApproPO.util import calc_dist_uni
from ApproPO.args import appropo_args

env = gym.make('FrozenMarsRoverEnvDynamic-v0')

# Fix Random seeds in all the underlying libraries
RANDOM_SEED = random.randint(0,1e+5)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
env.seed(RANDOM_SEED)
env.reset()

parser = argparse.ArgumentParser()
appropo_args(parser)
args = parser.parse_args()

def main():
    theta = np.zeros(2 + 8*8)
    theta[1] = 1
    if args.init_variable== 'constraint':
        theta[0] = _lambda = 0.6
    else:
        theta[0] = _lambda = 0.0

    stats = defaultdict(int)

    net = MLP(env)
    rl_oracle = RL_Oracle_Actor_Critic(env=env, net=net, args=args, theta=theta)
    policy = RCPOPolicy()
    shadow = RCPOPolicy(shadow=True)

    total_num_samples = 25000

    for episode in range(args.num_episodes):
        [rtn, episode_stats] = rl_oracle.learn_policy(n_traj=1, n_iter=args.rl_iter,\
                                                 normalize_theta=False, update=True)
        #with torch.no_grad():
        [shadow_exp_rtn, shadow_stats] =\
            rl_oracle.learn_policy(n_traj=args.check_traj, n_iter=args.rl_iter,\
                                    normalize_theta=False, update=False)

        stats['episode'] = episode
        stats['num_samples'] += sum(episode_stats['traj_len'])
        stats['num_trajs'] += len(episode_stats['traj_len'])
        stats['oracle_calls'] += 1
        stats['goals'] = np.average(episode_stats['goals'])
        episode_stats['_lambda'].append(_lambda)
        shadow_stats['_lambda'].append(_lambda)
        policy.add_response(episode_stats=episode_stats, episode=episode, rtn=rtn)
        shadow.add_response(episode_stats=shadow_stats, episode=episode, rtn=shadow_exp_rtn)

        _lambda = max(0, _lambda + args.lambda_lr*(-args.prob_failure-rtn[0]))

        theta[0] = _lambda
        rl_oracle.theta =  theta

        if episode % args.log_interval == 0:
            print('Episode {}\tDist to Uniform: {:.2f}\tReward: {:.2f}\t Prob Failure: {:.2f} \tGoal: {:.2f}\tlambda: {:.2f}'.format(episode, calc_dist_uni(rtn[2:]), rtn[1], rtn[0], sum(episode_stats['goals']), _lambda))

        if stats['num_samples'] > total_num_samples:
            break

    now = datetime.now() # current date and time
    date = now.strftime("%Y%m%d%H%M%S")
    df = pd.DataFrame(policy.stats,\
                  columns=np.hstack(['ep', 'traj', 'reward', 'prob_failure',\
                                     'obs', 'dist_uni', 'entropy', 'oracle_calls',\
                                 'cache_calls', 'traj_len', 'avg_traj_len', 'samples', 'lambda']))

    shadow_df = pd.DataFrame(shadow.stats,\
         columns=np.hstack(['ep', 'traj', 'reward', 'prob_failure',\
                            'obs', 'dist_uni', 'entropy', 'oracle_calls',\
                            'cache_calls', 'traj_len', 'avg_traj_len', 'samples', 'lamnda']))

    df.to_csv(f'{args.output}/rcpo_{args.name}.csv', index=False)
    shadow_df.to_csv(f'{args.output}/rcpo_{args.name}_avg_{total_num_samples}.csv', index=False)

if __name__ == '__main__':
    main()
