import numpy as np
np.set_printoptions(suppress=True)
import random
import torch
import argparse
import gym

from ApproPO.projection_oracle import ProjectionOracle
from ApproPO.rl_oracle_actor_critic import RL_Oracle as RL_Oracle_Actor_Critic
from ApproPO.nets import MLP
from ApproPO.envs import gym_frozenmarsrover
from ApproPO.envs.gym_frozenmarsrover.envs.maps import MAPS
from ApproPO.args import appropo_args
import ApproPO.solver as solver

# Fix Random seeds in all the underlying libraries
RANDOM_SEED = random.randint(0,1e+5)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f'Seed: {RANDOM_SEED}')

parser = argparse.ArgumentParser()
appropo_args(parser)
args = parser.parse_args()

def proj_basic(p):
    p = np.ndarray.copy(p)

    n_bins = 64
    bins = np.ones(n_bins)

    # We do not want the policy to have uniform visitation of the
    # risk zones
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

    if args.diversity:
        ratio = args.threshold
        if p[0] > 0: p[0] = 0
        elif p[0] < args.prob_failure: p[0] = args.prob_failure

        if p[1] < -0.17: p[1] = -0.17
        #elif p[1] < -0.17: p[1] = -0.17

        d = p[2:2+n_bins]-bins
        if np.linalg.norm(d) > ratio:
            p[2:2+n_bins] = (d/np.linalg.norm(d))*ratio+bins
    else:
        if p[0] > 0: p[0] = 0
        elif p[0] < args.prob_failure: p[0] = args.prob_failure

        if p[1] < -0.17: p[1] = -0.17
        #elif p[1] < -0.17: p[1] = -0.17
    return p

def main():
    theta = np.zeros(2 + 8*8)
    # setup environement
    env = gym.make('FrozenMarsRoverEnvDynamic-v0')
    env.reset()

    # setup rl oracle
    net = MLP(env)
    net = net.to(args.device)
    rl_oracle_generator = lambda: RL_Oracle_Actor_Critic(env=env, net=net, args=args, theta=theta)

    # setup proj oracle
    proj_oracle = ProjectionOracle(dim=theta.size, proj=proj_basic, args=args)

    solver.run(proj_oracle=proj_oracle, rl_oracle_generator=rl_oracle_generator, args=args)

if __name__ == "__main__":
    main()

