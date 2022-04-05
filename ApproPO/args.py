import torch
import random
import numpy as np
import argparse

RANDOM_SEED = random.randint(0,1e+5)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def appropo_args(parser):
    # Solver
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--cache_size", type=int, default=5)
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--print', default=True, action='store_true')
    parser.add_argument('--diversity', default=False, action='store_true')
    parser.add_argument('--threshold', type=float, default=0.12)
    parser.add_argument('--check_traj', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mx_size', type=int, default=20)
    parser.add_argument('--name', type=str, default='tmp')
    parser.add_argument('--init_variable', default='both', choices=['constraint', 'reward', 'both','None']) # use safety constraint same with rcpo theta[0]= 0.6

    # RL Oracle
    parser.add_argument("--rl_iter", type=int, default=300) # step
    parser.add_argument("--rl_traj", type=int, default=20)
    parser.add_argument("--rl_lr", type=float, default=1e-2)
    parser.add_argument("--entropy_coef", type=float, default=.001)
    parser.add_argument("--value_coef", type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', # discount factor
                        help='discount factor (default: 0.99)')
    # Projection Oracle
    parser.add_argument("--proj_lr", type=float, default=1)
    parser.add_argument("--olo_optim", choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--atol_cache', type=float, default=0.0)
    parser.add_argument('--atol_proj', type=float, default=1e-1)
    parser.add_argument('--rtol_cache', type=float, default=1e-05)
    parser.add_argument('--rtol_proj', type=float, default=1e-05)
    parser.add_argument('--prob_failure', type=float, default=-0.20)

    # RCPO
    parser.add_argument('--num_episodes', type=int, default=1000000)
    parser.add_argument('--lambda_lr', type=float, default=0.000025)
    parser.add_argument('--constraint', type=float, default=.20)
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')

    # Frozen Mars Rover Environment
    parser.add_argument('--map', choices=['4x4', '8x8'], default='8x8')
