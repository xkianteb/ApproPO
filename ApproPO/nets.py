import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class MLP(nn.Module):
    def __init__(self, env):
        super(MLP, self).__init__()
        try:
            # Note: env.reset().size == env.observation_space.n
            self.in_dim = env.observation_space.n
        except:
            self.in_dim = env.reset().size
            #self.in_dim = env.reset()[0].size

        self.out_dim = env.action_space.n
        self.eps = np.finfo(np.float32).eps.item()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        hidden_layer = 128
        self.affine1 = init_(nn.Linear(self.in_dim, hidden_layer))
        self.action_head = init_(nn.Linear(hidden_layer, self.out_dim))
        self.value_head = init_(nn.Linear(hidden_layer, 1))

    def forward(self, x):
        x = x.view(1, -1)
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return action_scores, state_values
