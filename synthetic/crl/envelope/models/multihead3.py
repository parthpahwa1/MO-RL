from __future__ import absolute_import, division, print_function
from re import X
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
        self.norm_layer = nn.LayerNorm(state_size)

        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 20)
        self.affine2 = nn.Linear((state_size + reward_size) * 20,
                                 (state_size + reward_size) * 20)
        self.affine3 = nn.Linear((state_size + reward_size) * 20,
                                 (state_size + reward_size) * 20)
        self.affine4 = nn.Linear((state_size + reward_size) * 20,
                                 (state_size + reward_size) * 20)
        self.Q = nn.Linear((state_size + reward_size) * 20,
                                 action_size * reward_size)
        
        self.affine5 = nn.Linear(action_size * reward_size + reward_size,
                                 (action_size * reward_size + reward_size) * 20)
        self.affine6 = nn.Linear((action_size * reward_size + reward_size) * 20,
                                 (action_size * reward_size + reward_size) * 20)
        self.affine7 = nn.Linear((action_size * reward_size + reward_size) * 20,
                                (action_size * reward_size + reward_size) * 20)
        self.affine8 = nn.Linear((action_size * reward_size + reward_size) * 20,
                                 (action_size * reward_size + reward_size) * 20)

        self.q_scalar = nn.Linear((action_size * reward_size + reward_size) * 20,
                                 action_size)

    def forward(self, state, preference):
        # state = self.norm_layer(state)
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        Q = self.Q(x)
        
        intermediate = torch.cat((Q, preference), dim=1)
        
        Q = Q.view(Q.size(0), self.action_size, self.reward_size)
        
        x = F.relu(self.affine5(intermediate))
        x = F.relu(self.affine6(x))
        x = F.relu(self.affine7(x))
        x = F.relu(self.affine8(x))
        
        q_scalar = self.q_scalar(x)

        hq = q_scalar.detach().max(dim=1)[0]

        return hq, Q, q_scalar
