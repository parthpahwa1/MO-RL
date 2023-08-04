from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class EnvelopeLinearCQN(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(EnvelopeLinearCQN, self).__init__()

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
        self.affine5 = nn.Linear((state_size + reward_size) * 32,
                                 action_size * reward_size)
        self.q_scalar = nn.Linear((state_size + reward_size) * 32 + action_size * reward_size + reward_size,
                                 action_size)

    def forward(self, state, preference):
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        Q = self.affine5(x)
        intermediate = torch.cat((Q, preference), dim=1)
        
        Q = Q.view(Q.size(0), self.action_size, self.reward_size)
        
        q_scalar = self.q_scalar(torch.cat((x, intermediate), dim=1))

        hq = q_scalar.detach().max(dim=1)[0]

        return hq, Q, q_scalar
