import numpy as np
import sys
from six import StringIO, b
import torch
from gym import spaces
import random

from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv
from ApproPO.envs.gym_frozenmarsrover.envs.maps import MAPS

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class FrozenMarsRoverEnv(DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=False, type=None):
        self.reward_type = type
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        #if is_slippery:
                        if False:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))


        super(FrozenMarsRoverEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile

    def _state_repr(self, location):
        [risk_zone, goal] = self.get_env_attributes()

        if self.observation_space.n == 16 or self.observation_space.n == 64: # Medium
            grid = np.zeros(self.observation_space.n)
            grid[risk_zone] = 1.0
            grid[goal] = 2.0
            grid[location] = 3.0
            self.goal = goal
            self.risk_zone = risk_zone
        elif  self.observation_space.n == 900: # Large
            grid = np.zeros((1, 1, 30, 30))
            for (x, y) in risk_zone:
                grid[0, 0, x, y] = 1

            (x, y) = goal
            grid[0, 0, x, y] = 2
            self.goal = 148
            self.risk_zone = risk_zone

            tmp_grid = np.zeros((30, 30))
            tmp_grid = tmp_grid.flatten()
            tmp_grid[location] =  3
            tmp_grid = tmp_grid.reshape((30, 30))
            location = np.where(tmp_grid == 3)
            loc = (location[0][0], location[1][0])

            grid[0, 0, loc] = 3
        return np.array(grid / 3.0)

    def get_env_attributes(self):
        if self.observation_space.n == 16: # Small
            risk_zones = [5, 7, 11, 12]
            goal = 15
        elif self.observation_space.n == 64: # Medium
            risk_zones = [11, 14, 35, 41, 42, 48, 51, 53, 58]
            goal = 63
        elif  self.observation_space.n == 900: # Large
            risk_zones = [(1, 20),(2, 7), (2, 11), (2, 17), (2, 19), (3, 13),\
                          (4, 10), (4, 16), (4, 24), (5, 6), (5, 20), (6, 12),\
                          (7, 17), (7, 22), (8, 9), (8, 14), (9, 7), (9, 18),\
                          (10, 22), (12, 12), (12, 19), (13, 15), (15, 20),\
                          (16, 12), (17, 16), (17, 22), (19, 5), (22, 10), (28, 19)]
            goal = (4, 28)
        else:
            raise Exception("Unknown Demnension")
        self.metadata['goal'] = goal
        self.metadata['risk_zones'] = risk_zones
        return risk_zones, goal

    def step(self, action):
        # The environment is stochastic: with probability δ = 0:05 the agent’s action is perturbed to a random action
        if random.uniform(0,1) < 0.05 and self.reward_type == 'rcpo':
            action = random.randint(0,3)

        state_idx, reward, done, info = DiscreteEnv.step(self, action)

        if done and reward == 0: # Failure
            reward = 0
        elif done and state_idx == self.goal:  # Win
            reward = 0
        else: # Move to another case
            reward = -0.01

        goal = False
        if (done and state_idx == self.goal):
            goal=True

        #constraint = (state_idx in self.risk_zone)
        constraint = int(done and not goal)

        state = self._state_repr(state_idx)
        return state, reward, done, {'goal': goal, 'constraint': constraint}

    def reset(self):
        state = DiscreteEnv.reset(self)
        return self._state_repr(state)

