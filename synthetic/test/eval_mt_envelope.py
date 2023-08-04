from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE

import gym
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv
import mo_gym



parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='ft', metavar='ENVNAME',
                    help='environment to train on (default: tf): ft | ft5 | ft7')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.995, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# PLOT
parser.add_argument('--pltmap', default=False, action='store_true',
                    help='plot deep sea treasure map')
parser.add_argument('--pltpareto', default=True, action='store_true',
                    help='plot pareto frontier')
parser.add_argument('--pltcontrol', default=True, action='store_true',
                    help='plot control curve')
parser.add_argument('--pltdemo', default=False, action='store_true',
                    help='plot demo')
# LOG & SAVING
parser.add_argument('--alpha', type=float, default=5000, metavar='ALPHA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0., metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


model_loc = '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_449.pkl'
model_loc = '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_749.pkl'
model_loc = '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_1049.pkl'
model_loc = '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_1499.pkl'
model_loc = '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_2249.pkl'
model_loc = '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_2999.pkl'


model_list = {
449 : '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_449.pkl',
749 : '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_749.pkl',
1049 : '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_1049.pkl',
1499 : '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_1499.pkl',
2249 : '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_2249.pkl',
2999 : '../saved_models/multihead2_Q_log_ft_2022_06_21_envelope_eps_2999.pkl'
}


env = gym.make('mo-mountaincar-v0')

# generate an agent for plotting
agent = None

from crl.envelope.meta import MetaAgent
# import pandas as pd

vis = visdom.Visdom()
assert vis.check_connection()

model_loc = model_list[1499]
model = torch.load(model_loc)
agent = MetaAgent(model, args, is_train=False)

opt_x = []
opt_y = []
q_x = []
q_y = []
act_x = []
act_y = []

for i in range(500):
    w = np.random.randn(3)
    w[2] = 0
    w = np.abs(w) / np.linalg.norm(w, ord=1)

    # w = np.random.dirichlet(np.ones(2))
    w_e = w / np.linalg.norm(w, ord=2)
    
    # hq, Q, q = agent.predict(torch.from_numpy(w).type(FloatTensor))
    
    # arr_indx = q.max(dim=1)[1]
    # qc = Q[0].detach().numpy().dot(w).max() * w_e
    
    ttrw = np.array([0.0, 0.0, 0.0])
    terminal = False
    state = env.reset()
    cnt = 0
    w = FloatTensor(w)
    reward_list = []
    for j in range(50):
        env.reset()
        cnt = 0
        state = env.reset()
        terminal = False
        probe =  FloatTensor(w)
        ttrw = np.array([0.0, 0.0, 0.0])
        while not terminal:
            action = agent.act(state, probe)
            next_state, reward, terminal, _ = env.step(action)
            state = next_state
            next_preference = FloatTensor(w)
            reward = _['reward']

            if next_state[0] - state[0] > 0 and action == 2: 
                reward += 0.5
            if next_state[0] - state[0] < 0 and action == 0: 
                reward += 0.5
            
            if cnt > 300:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1
        
        ttrw_w = w.dot(FloatTensor(ttrw))*w_e

        reward_list.append(np.array(ttrw_w))

    ttrw_w = np.mean(np.array(reward_list), axis=0)

    # q_x.append(qc[0].tolist())
    # q_y.append(qc[2].tolist())
    act_x.append(ttrw_w[0].tolist())
    act_y.append(ttrw_w[1].tolist())


act_opt = dict(x=act_x,
                y=act_y,
                mode="markers",
                type='custom',
                marker=dict(
                    symbol="circle",
                    size=1),
                name='policy')

# q_opt = dict(x=q_x,
#                 y=q_y,
#                 mode="markers",
#                 type='custom',
#                 marker=dict(
#                     symbol="circle",
#                     size=1),
#                 name='predicted')


layout_opt = dict(title="Mountain Car: Envelope Q-learning Recovered CCS",
    xaxis=dict(title='Time penalty'),
    yaxis=dict(title='Reverse Penalty'))

vis._send({'data': [act_opt], 'layout': layout_opt})


# df = pd.DataFrame(columns=['Train Episode', 'Steps to terminate', 'Reward', 'Left Action', 'Right Action', 'No Action', 'Time Penalty', 'Left penalty', 'Right penalty'])

# for key in model_list.keys():
#     model_loc = model_list[key]
#     model = torch.load(model_loc)
#     agent = MetaAgent(model, args, is_train=False)
    
#     probe_list = np.array([[0.9, 0.05, 0.05], [0.9, 0.1, 0.0], [0.9, 0, 0.1], [0.5, 0., 0.5], [0.5, 0.5, 0 ], [0., 0.5, 0.5], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

#     for probe in probe_list:
#         steps_to_terminate = []
#         action_count_tracker = {
#             0: [],
#             1: [],
#             2: []
#         }
#         reward_list = []
#         for i in range(100):
#             w_e = probe
#             w = w_e

#             action_count= {
#                 0: 0,
#                 1: 0,
#                 2: 0
#             }


#             env.reset()
#             cnt = 0
#             state = env.reset()
#             terminal = False
#             probe =  FloatTensor(probe)
#             ttrw = np.array([0.0, 0.0, 0.0])
#             while not terminal:
#                 action = agent.act(state, probe)
#                 next_state, reward, terminal, _ = env.step(action)
#                 state = next_state
#                 next_preference = FloatTensor(probe)
#                 reward = _['reward']

#                 if next_state[0] - state[0] > 0 and action == 2: 
#                     reward += 0.5
#                 if next_state[0] - state[0] < 0 and action == 0: 
#                     reward += 0.5

#                 action_count[action] += 1
                
#                 if cnt > 300:
#                     terminal = True
#                 ttrw = ttrw + reward * np.power(args.gamma, cnt)
#                 cnt += 1
            
#             ttrw_w = w.dot(FloatTensor(ttrw))

#             steps_to_terminate.append(cnt)
#             reward_list.append(ttrw_w)
#             action_count_tracker[0].append(action_count[0])
#             action_count_tracker[1].append(action_count[1])
#             action_count_tracker[2].append(action_count[2])
        
#         print('Reward:', np.mean(np.array(reward_list)))
#         print ('steps_to_terminate:' , np.mean(np.array(steps_to_terminate)))
#         print ('Left acceleration mean:',np.mean(np.array(action_count_tracker[0])))
#         print ('Do not accelerate:',np.mean(np.array(action_count_tracker[1])))
#         print ('Right acceleratio:',np.mean(np.array(action_count_tracker[2])))
#         print ('Time Penalty, Left acceleration penalty, Right acceleration penalty:', probe)

#         data = [key, np.mean(np.array(steps_to_terminate)), np.mean(np.array(reward_list)), np.mean(np.array(action_count_tracker[0])), np.mean(np.array(action_count_tracker[2])) ,np.mean(np.array(action_count_tracker[1])), probe[0], probe[1], probe[2]]
        
#         data = np.array(data).reshape(1, -1)
#         df_temp = pd.DataFrame(data, columns=['Train Episode', 'Steps to terminate', 'Reward', 'Left Action', 'Right Action', 'No Action', 'Time Penalty', 'Left penalty', 'Right penalty'])
        
#         df = pd.concat([df, df_temp])      
#         print('-----------------------------------------------------------------')

# df.to_csv('experiment_mt_envelope.csv')
