from __future__ import absolute_import, division, print_function
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init

from torch.distributions.categorical import Categorical

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class A2C(nn.Module):
    def __init__(self, state_size, action_size, reward_size, parameter_size):
        super(A2C, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 16)
        self.affine2 = nn.Linear((state_size + reward_size) * 16,
                                 (state_size + reward_size) * 32)
        self.affine3 = nn.Linear((state_size + reward_size) * 32,
                                 (state_size + reward_size) * 64)
        self.affine4 = nn.Linear((state_size + reward_size) * 64,
                                 (state_size + reward_size) * 32)
        
        self.actor = nn.Linear((state_size + reward_size) * 32, action_size)
        self.critic_vector = nn.Linear((state_size + reward_size) * 32, reward_size)
        self.critic_scalar = nn.Linear((state_size + reward_size) * 32, 1)


    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))

        policy = self.actor(x)
        value = [self.critic_scalar(x), self.critic_vector(x)]
        
        return policy, value
