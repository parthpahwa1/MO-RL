from __future__ import absolute_import, division, print_function
from os import stat
import random
from timeit import repeat
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
import torch.nn as nn
from torch.distributions import Categorical
from .models import A2C

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):

    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model:A2C, args, is_train=False):
        self.device = torch.device('cuda' if use_cuda else 'cpu')


        self.model = model
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = True
        

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num
        self.clip_grad_norm = 1
        self.entropy_coef = 0.05
        self.entropy_coef_delta = 0

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
    
    def random_choice_prob_index(self, p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)
    
    def get_action(self, state, preference):
        state = torch.from_numpy(state).to(self.device)
        policy, value = self.model(Variable(state.unsqueeze(0)), Variable(preference.unsqueeze(0)))
        

        policy = F.softmax(policy, dim=-1).data.cpu().numpy()[0]

        action = np.random.choice([x for x in range(0,len(policy))], p=policy)

        return action

    def get_value_for_state(self, state, preference):
        state = torch.from_numpy(state).to(self.device)

        policy, value = self.model(Variable(state.unsqueeze(0)), Variable(preference.unsqueeze(0)))
        
        return value[0], value[1], policy

    def forward_transition(self, state, next_state, preference, next_preference):
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        preference = torch.from_numpy(preference).to(self.device)
        next_preference = torch.from_numpy(next_preference).to(self.device)

        # print('forward_transition:', state.shape, preference.shape)
        # print('squeeze forward_transition:', Variable(state.unsqueeze(0)).shape, Variable(preference.unsqueeze(0)).shape)
        policy, value = self.model(Variable(state), Variable(preference))
        _, next_value = self.model(Variable(next_state), Variable(next_preference))

        value_scalar = value[0]
        value_vector = value[1]
        value_scalar = value_scalar.data.cpu().numpy().squeeze()
        value_vector = value_vector.data.cpu().numpy().squeeze()
        
        next_value_scalar = next_value[0]
        next_value_vector = next_value[1]
        next_value_scalar = next_value_scalar.data.cpu().numpy().squeeze()
        next_value_vector = next_value_vector.data.cpu().numpy().squeeze()

        return [value_scalar, value_vector], [next_value_scalar, next_value_vector], policy

    def learn(self, episode_data):
        curr_state_batch = episode_data['curr_state']
        curr_preference_batch = episode_data['curr_preference']

        next_state_batch = episode_data['next_state']
        next_preference_batch = episode_data['next_preference']

        target_scalar_batch = episode_data['target_scalar']
        target_vector_batch = episode_data['target_vector']
        
        action_batch = episode_data['action']
        
        advantage_batch = episode_data['advantage']
        advantage_batch_vector = episode_data['advantage_vector']

        with torch.no_grad():
            s_batch = torch.FloatTensor(curr_state_batch).to(self.device)
            next_s_batch = torch.FloatTensor(next_state_batch).to(self.device)
            
            curr_w_batch = torch.FloatTensor(curr_preference_batch).to(self.device)
            next_w_batch = torch.FloatTensor(next_preference_batch).to(self.device)
            
            target_scalar_batch = torch.FloatTensor(target_scalar_batch).to(self.device)
            target_vector_batch = torch.FloatTensor(target_vector_batch).to(self.device)

            action_batch = torch.LongTensor(action_batch).to(self.device)
            advantage_batch = torch.FloatTensor(advantage_batch).to(self.device)
            advantage_batch_vector = torch.FloatTensor(advantage_batch_vector).to(self.device)


        # for multiply advantage
        policy, value = self.model(s_batch, curr_w_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        # # Actor loss
        # print(wlse_w.shape, action_batch.shape, advantage_batch.shape)
        actor_loss = -m.log_prob(action_batch) * (advantage_batch.detach())

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        critic_loss = 0.5*F.mse_loss(value[0], target_scalar_batch.unsqueeze(1))
        critic_loss += 0.5*F.mse_loss(value[1], target_vector_batch)

        self.optimizer.zero_grad()

        loss = actor_loss.mean() + critic_loss - self.entropy_coef * entropy.mean()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        
        self.optimizer.step()

    def reset(self):
        if self.epsilon_decay:
            self.entropy_coef -=self.entropy_coef_delta

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))
