from __future__ import absolute_import, division, print_function
from datetime import datetime
from re import X
from matplotlib.pyplot import axis
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
from .exemplar import Exemplar

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

    def __init__(self, model, exemplar_exploration:Exemplar, args, is_train=False):
        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.2) / (1500)

        self.exploration_alpha = 0.01
        self.exemplar_update_freq = 100
        self.exploration_delta = (self.exploration_alpha - 0.01) / (args.episode_num)
        

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        self.beta_uplim = 1.00
        self.clip_grad_norm = 1
        self.alpha = args.alpha

        self.trans_mem = deque()
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd', 'w', 'w_'])
        self.priority_mem = deque()

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.update_count = 0
        self.update_freq = args.update_freq
        
        self.min_loss1 = 9999
        self.min_loss2 = 9999
        self.min_loss_mean = 9999

        self.save_loc = './saved_models/'
        self.save_file_name = 'multihead3_Q_' + args.env_name + '_' + str(datetime.today().strftime("%Y_%m_%d"))

        self.exemplar_exploration = exemplar_exploration
        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

        self.weight_loss_list = {
            'w1': [],
            'w2': [],
            'loss': []
        }

    def generate_neighbours(self, preference, next_preference, weight_num):
        preference_neigbor = preference.clone().detach()
        next_preference_neigbor = next_preference.clone().detach()
        
        new_preference = np.random.randn(preference.shape[0] * (weight_num-1), preference.shape[1])
        new_preference = np.abs(new_preference)/np.linalg.norm(new_preference, ord=1,axis=1)[:,None]

        cov = np.identity(new_preference.shape[1])*0.0001
        new_next_preference = np.array([np.random.multivariate_normal(tensor, cov, 1)[0] for tensor in new_preference])

        new_next_preference[new_next_preference < 0] = 0
        new_next_preference += 1e-5
        new_next_preference = new_next_preference/np.sum(new_next_preference, axis=1)[:,None]

        preference_neigbor = torch.cat((preference_neigbor, FloatTensor(new_preference)), dim=0)
        next_preference_neigbor = torch.cat((next_preference_neigbor, FloatTensor(new_next_preference)), dim=0)
        
        return preference_neigbor, next_preference_neigbor


    def generate_neighbours2(self, preference, next_preference, weight_num):
        preference_neigbor = preference.clone().detach()
        next_preference_neigbor = next_preference.clone().detach()

        for i in range(weight_num-1):
            repeat = torch.tensor(())
            repeat_next = torch.tensor(())
            
            for tensor in preference:
                tensor = tensor.numpy()
                
                # if i % 2 == 0:
                #     tensor = FloatTensor(np.random.dirichlet(np.random.uniform(size=tensor.shape[0])/2)).unsqueeze(0)
                # else: 
                tensor = np.random.randn(len(tensor))
                tensor = np.abs(tensor)/np.linalg.norm(tensor, ord=1)
                tensor = FloatTensor(tensor).unsqueeze(0)

                # tensor = FloatTensor(np.random.dirichlet(np.ones(shape=tensor.shape))).unsqueeze(0)
                
                repeat = torch.cat((repeat, tensor), dim=0)
                next_tensor = tensor.clone().detach().numpy()[0] + 1e-5
                next_tensor = FloatTensor(np.random.dirichlet(self.alpha*next_tensor)).unsqueeze(0)
                # next_tensor = next_tensor/sum(next_tensor)
                repeat_next = torch.cat((repeat_next, next_tensor), dim=0)
                # tensor = tensor.numpy()[0]
                # tensor = tensor + 1e-6
                # repeat_next = torch.cat((repeat_next, FloatTensor(np.random.dirichlet(self.alpha*tensor)).unsqueeze(0)), dim=0)
            
            preference_neigbor = torch.cat((preference_neigbor, repeat))
            next_preference_neigbor = torch.cat((next_preference_neigbor, repeat_next))
        
        return preference_neigbor, next_preference_neigbor

    def act(self, state, preference=None, greedy=False):
        state = torch.from_numpy(state).type(FloatTensor)

        _, Q, q_scalar = self.model_(
            Variable(state.unsqueeze(0)),
            Variable(preference.unsqueeze(0)))

        action = q_scalar.max(1)[1].data.cpu().numpy()
        action = int(action[0])

        if greedy == True:
            return action, Q, q_scalar

        eps = np.random.uniform(high=self.epsilon)
        eps = self.epsilon

        if self.is_train and (len(self.trans_mem) < self.batch_size or torch.rand(1)[0] < eps):
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
            p += abs((wR + self.gamma * hQ[0, action_next] - Q_val)).mean()
        
        else:
            if self.epsilon_decay and self.epsilon > 0.1:
                self.epsilon -= self.epsilon_delta

                if self.epsilon < 0:
                    self.epsilon = 0.1
                
            p = np.exp(abs(wr - q_scalar))
            p += abs((wR - Q_val)).mean()
        
        
        p = p.detach().numpy()
        p =np.nan_to_num(p)
        p += 1e-5
        self.priority_mem.append(
            p
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
            
            # w_batch = batchify(map(lambda x: x.w, minibatch))
            # w_batch = Variable(torch.stack(w_batch), requires_grad=False).type(FloatTensor)

            w_batch = list(map(lambda x: x.w, minibatch))
            w_batch = Variable(torch.stack(w_batch), requires_grad=False).type(FloatTensor)
            next_w_batch = list(map(lambda x: x.w_, minibatch))
            next_w_batch = Variable(torch.stack(next_w_batch), requires_grad=False).type(FloatTensor)
            w_batch, next_w_batch = self.generate_neighbours(w_batch, next_w_batch, self.weight_num)

            # next_w_batch = batchify(map(lambda x: x.w_, minibatch))
            # next_w_batch = Variable(torch.stack(next_w_batch), requires_grad=False).type(FloatTensor)
            
            # next_w_batch = self.generate_neighbours(next_w_batch, self.weight_num)

            
            
            # Current State
            __, Q, Q_scalar = self.model_(Variable(torch.cat(state_batch, dim=0)), Variable(w_batch))
            
            # Next State
            _, DQ, DQ_scalar = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False), Variable(next_w_batch, requires_grad=False))
            _, tmpQ, tmpQ_scalar = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False), Variable(next_w_batch, requires_grad=False))

            action_next_state = tmpQ_scalar.max(dim=1)[1]

            
            HQ = DQ.gather(1, action_next_state.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()
            HQ_scalar = DQ_scalar.gather(1, action_next_state.unsqueeze(dim=1)).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch)
            
            exploration_reward = -1*self.exploration_alpha * torch.log(self.exemplar_exploration.get_prob(torch.cat((torch.cat(state_batch, dim=0), w_batch), dim=1)))
            exploration_reward = torch.clamp(exploration_reward, 0, 1)
            
            # exploration_reward = -1*self.exploration_alpha * torch.log(self.exemplar_exploration.get_prob(w_batch))

            # print(torch.max(exploration_reward))

            scalarized_reward_batch = torch.bmm(w_batch.unsqueeze(1), torch.cat(reward_batch, dim=0).unsqueeze(2)).squeeze()
            scalarized_reward_batch += exploration_reward
            
            # Get transition values
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num, reward_size).type(FloatTensor))
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]

                Tau_Q += Variable(torch.cat(reward_batch, dim=0))

                # Tau_Q += Variable(torch.add(torch.cat(reward_batch, dim=0),exploration_reward[:,None]/w_batch.shape[1] ))

                Tau_Q_scalar = Variable(torch.zeros(self.batch_size * self.weight_num).type(FloatTensor))
                Tau_Q_scalar[nontmlmask] = self.gamma * HQ_scalar[nontmlmask]
                Tau_Q_scalar += Variable(scalarized_reward_batch)

            actions = Variable(torch.cat(action_batch, dim=0))

            Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))).view(-1, reward_size)
            Tau_Q = Tau_Q.view(-1, reward_size)
            
            # w1 = 0.6
            # w2 = 0.4
            # huber_loss = torch.nn.HuberLoss()

            loss1 = F.mse_loss(Q_scalar.gather(1, actions.unsqueeze(dim=1)), Tau_Q_scalar.view(-1).unsqueeze(dim=1))*1.25
            loss2 = F.mse_loss(Q.view(-1), Tau_Q.view(-1))*0.05
            

            # loss1 = huber_loss(Q_scalar.gather(1, actions.unsqueeze(dim=1)), Tau_Q_scalar.view(-1).unsqueeze(dim=1))*4
            # loss2 = huber_loss(Q.view(-1), Tau_Q.view(-1))*0.4

            w1 = torch.exp(3*loss1)
            w2 = torch.exp(3*loss2)

            
            sum_w1_w2 = w1 + w2

            self.weight_loss_list['w1'].append((w1/sum_w1_w2).item())
            self.weight_loss_list['w2'].append((w2/sum_w1_w2).item())
            
            if self.update_count % self.update_freq == 0:
                print('exploration_reward', torch.mean(exploration_reward), torch.max(exploration_reward), torch.min(exploration_reward), self.exploration_alpha)
                print('loss1:', loss1.data, 'loss2:', loss2.data,  'w1:', (w1/sum_w1_w2).data, 'w2:', (w2/sum_w1_w2).data)

                
            # print('loss q scalar:', loss1, 'loss Q:', loss2, 'w1', w1, 'w2', w2)
            # if self.update_count > 2000:
            #     if loss1.data < self.min_loss1 and loss1.data < 0.01:
            #         self.min_loss1 = loss1.data
            #         self.save(self.save_loc, self.save_file_name+"_min_loss1_"+str(loss1.data)+"_count_"+str(self.update_count))

            #     elif loss2.data < self.min_loss2 and loss2.data < 0.07:
            #         self.min_loss2 = loss2.data
            #         self.save(self.save_loc, self.save_file_name+"_min_loss2_"+str(loss2.data)+"_count_"+str(self.update_count))

            
            # loss = (w1*loss1 + w2*loss2)/sum_w1_w2
            loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            
            # for param in self.model_.parameters():
            #     param.grad.data.clamp_(-1, 1)
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            self.weight_loss_list['loss'].append(loss.item())

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            if (self.update_count % self.exemplar_update_freq == 0) and  self.exploration_alpha > 0.005:
                self.exploration_alpha -= self.exploration_delta

                if self.exploration_alpha < 0:
                    self.exploration_alpha = 0.005

            # if self.update_count % self.exemplar_update_freq == 0:
            
            num_batches = 5
            exemplar_batch_size = self.batch_size

            
            # num_batches = 1
            for i in range(num_batches):
                minibatch = self.sample(self.trans_mem, self.priority_mem, exemplar_batch_size)
                batchify = lambda x: list(x)
                state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
                w_batch = batchify(map(lambda x: x.w, minibatch))
                w_batch = Variable(torch.stack(w_batch), requires_grad=False).type(FloatTensor)

                # index_list = np.random.randint(0, exemplar_batch_size, size=self.batch_size)

                sample1 = torch.cat((torch.cat(state_batch, dim=0), w_batch), dim=1)
                # sample1 = w_batch[index_list]
                
                # sample1 = torch.cat((torch.cat(state_batch, dim=0)[0:self.batch_size], w_batch[0:self.batch_size]), dim=1)

                positive = sample1[0:int(sample1.shape[0]/2)]
                negative = sample1[int(sample1.shape[0]/2):]

                sample1 = torch.cat((positive, positive), axis=0)
                sample2 = torch.cat((positive, negative), axis=0)

                target = torch.cat((torch.ones((positive.shape[0], 1)), torch.zeros((negative.shape[0],1))))

                exploration_loss = self.exemplar_exploration.update(sample1, sample2, target)

            return [loss.data, exploration_loss[0]]

        return [0.0, 0.0]

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
