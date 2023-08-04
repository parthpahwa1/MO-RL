from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

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

    def __init__(self, model, args, is_train=False):
        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.02) / (args.episode_num-500)

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        self.beta_uplim      = 1.00
        self.clip_grad_norm = 1
        self.alpha = args.alpha

        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd', 'w', 'w_'])
        self.priority_mem = deque()

        self.w_kept = None

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def generate_neighbours(self, preference, next_preference, weight_num):
        preference_neigbor = preference.clone().detach()
        next_preference_neigbor = next_preference.clone().detach()


        for i in range(weight_num-1):
            repeat = torch.tensor(())
            repeat_next = torch.tensor(())
            
            for tensor in preference:
                tensor = tensor.numpy()
                
                tensor = FloatTensor(np.random.dirichlet(np.random.uniform(size=tensor.shape[0]))).unsqueeze(0)
                repeat = torch.cat((repeat, tensor), dim=0)
                
                tensor = tensor.numpy()[0]
                tensor = tensor + 1e-6
                repeat_next = torch.cat((repeat_next, FloatTensor(np.random.dirichlet(self.alpha*tensor)).unsqueeze(0)), dim=0)
            
            preference_neigbor = torch.cat((preference_neigbor, repeat))
            next_preference_neigbor = torch.cat((next_preference_neigbor, repeat_next))
        
        return preference_neigbor, next_preference_neigbor

    def act(self, state, preference=None, greedy=False):
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept
        # print(preference)        
        state = torch.from_numpy(state).type(FloatTensor)

        _, Q, q_scalar = self.model_(
            Variable(state.unsqueeze(0)),
            Variable(preference.unsqueeze(0)))

        action = q_scalar.max(1)[1].data.cpu().numpy()
        action = int(action[0])

        if greedy == True:
            return action, Q, q_scalar

        if self.is_train and (len(self.trans_mem) < self.batch_size or torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.model_.action_size, 1)[0]
            action = int(action)

        return action

    def memorize(self, state, action, next_state, reward, terminal, preference, next_preference):
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal, # terminal
            preference, # w
            next_preference))  # w_

        state = torch.from_numpy(state).type(FloatTensor)

        _, Q, q_scalar = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))

        q_scalar = q_scalar[0, action].data
        Q_val = Q[0, action]
        
        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        wR = FloatTensor(reward)
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            hq, hQ, hq_scalar = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False))
            hq = hq.data[0]
            action_next = hq_scalar.max(dim=1)[1]
            p = abs(wr + self.gamma * hq - q_scalar)
            # print(p)
            # print(abs((wR + self.gamma * hQ[0, action] - Q_val)).max())
            # print('-0-----------------')
            p += abs((wR + self.gamma * hQ[0, action_next] - Q_val)).max()
        
        else:
            self.w_kept = None
            if self.epsilon_decay and self.epsilon > 0.01:
                self.epsilon -= self.epsilon_delta

                if self.epsilon < 0:
                    self.epsilon = 0.01
                
            p = abs(wr - q_scalar)
            p += abs((wR - Q_val)).max()
        
        p += 1e-5

        self.priority_mem.append(
            p.detach().numpy()
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self):
        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            reward_size = self.model_.reward_size

            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))
            
            w_batch = np.random.randn(self.weight_num, reward_size)
            w_batch = np.abs(w_batch) / \
                      np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
            w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)

            
            
            # Current State
            __, Q, Q_scalar = self.model_(Variable(torch.cat(state_batch, dim=0)), Variable(w_batch))
            
            # Next State
            _, DQ, DQ_scalar = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False), Variable(w_batch, requires_grad=False))
            _, tmpQ, tmpQ_scalar = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False), Variable(w_batch, requires_grad=False))

            action_next_state = tmpQ_scalar.max(dim=1)[1]

            
            HQ = DQ.gather(1, action_next_state.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()
            HQ_scalar = DQ_scalar.gather(1, action_next_state.unsqueeze(dim=1)).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch)
            
            scalarized_reward_batch = torch.bmm(w_batch.unsqueeze(1), torch.cat(reward_batch, dim=0).unsqueeze(2)).squeeze()
            
            # Get transition values
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num, reward_size).type(FloatTensor))
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                Tau_Q += Variable(torch.cat(reward_batch, dim=0))

                Tau_Q_scalar = Variable(torch.zeros(self.batch_size * self.weight_num).type(FloatTensor))
                Tau_Q_scalar[nontmlmask] = self.gamma * HQ_scalar[nontmlmask]
                Tau_Q_scalar += Variable(scalarized_reward_batch)

            actions = Variable(torch.cat(action_batch, dim=0))

            Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))).view(-1, reward_size)
            Tau_Q = Tau_Q.view(-1, reward_size)
            
            # w1 = torch.exp(3*F.mse_loss(Q_scalar.gather(1, actions.unsqueeze(dim=1)), Tau_Q_scalar.view(-1).unsqueeze(dim=1)))
            # w2 = torch.exp(3*F.mse_loss(Q.view(-1), Tau_Q.view(-1)))
            
            w1 = 0.5
            w2 = 0.5

            sum_w1_w2 = w1 + w2
            loss = w1*F.mse_loss(Q_scalar.gather(1, actions.unsqueeze(dim=1)), Tau_Q_scalar.view(-1).unsqueeze(dim=1))/sum_w1_w2
            loss += w2*F.mse_loss(Q.view(-1), Tau_Q.view(-1))/sum_w1_w2

            self.optimizer.zero_grad()
            loss.backward()
            
            # for param in self.model_.parameters():
            #     param.grad.data.clamp_(-1, 1)
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.clip_grad_norm)

            self.optimizer.step()
            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            return loss.data

        return 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay and self.epsilon > 0.01:
            self.epsilon -= self.epsilon_delta

            if self.epsilon < 0:
                self.epsilon = 0.01

    def predict(self, probe, state=[]):
        if len(state) == 0:
            return self.model(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
                            Variable(probe.unsqueeze(0), requires_grad=False))
        else :
            return self.model(Variable(FloatTensor(state).unsqueeze(0), requires_grad=False),
                            Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))
