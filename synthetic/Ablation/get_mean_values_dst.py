from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
from torch.autograd import Variable
import time as Timer
import math
import numpy as np

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
model_loc = './dst/multihead3_Q_without_exemplar_dst_2022_06_24_eps_2000.pkl'

sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on (default: dst)')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
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
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
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
parser.add_argument('--alpha', type=float, default=5000, metavar='ALPHA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()
vis = visdom.Visdom()

assert vis.check_connection()

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Add data
alpha = args.alpha

gamma = args.gamma

time = [-1, -3, -5, -7, -8, -9, -13, -14, -17, -19]
# treasure = [0.1, 2.8, 5.2, 7.3, 8.2, 9.0, 11.5, 12.1, 13.5, 14.2]
treasure = [0.7, 8.2, 11.5, 14., 15.1, 16.1, 19.6, 20.3, 22.4, 23.7]
# time	 = [ -1, -3, -5, -7, -8, -9]
# treasure = [0.1, 2.8, 5.2, 7.3, 8.2, 9.0]

# apply gamma
dis_time = (-(1 - np.power(gamma, -np.asarray(time))) / (1 - gamma)).tolist()
dis_treasure = (np.power(gamma, -np.asarray(time) - 1) * np.asarray(treasure)).tolist()

def find_in(A, B, base=0):
    # base = 0: tolerance w.r.t. A
    # base = 1: tolerance w.r.t. B
    # base = 2: no tolerance
    cnt = 0.0
    for a in A:
        for b in B:
            if base == 0:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(a):
                  cnt += 1.0
                  break
            elif base == 1:
              if np.linalg.norm(a - b, ord=1) < 0.20*np.linalg.norm(b):
                  cnt += 1.0
                  break
            elif base == 2:
              if np.linalg.norm(a - b, ord=1) < 0.3:
                  cnt += 1.0
                  break
    return cnt / len(A)

################# Control Frontier #################

if args.pltcontrol:

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    agent = None


    from crl.envelope.meta_dst import MetaAgent
    model = torch.load(model_loc)
    agent = MetaAgent(model, None, args, is_train=False)

    # compute opt
    opt_x = []
    opt_y = []
    q_x = []
    q_y = []
    act_x = []
    act_y = []
    real_sol = np.stack((dis_treasure, dis_time))

    policy_loss = np.inf
    predict_loss = np.inf

if args.pltpareto:

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # generate an agent for plotting
    agent = None
    from crl.envelope.meta_dst import MetaAgent
    model = torch.load(model_loc)
    agent = MetaAgent(model, None, args, is_train=False)

    policy_f1 = []
    for j in range(10):
        # compute recovered Pareto
        act_x = []
        act_y = []

        # predicted solution
        pred_x = []
        pred_y = []
        pred = []

        
        for i in range(2000):
            w = np.random.randn(2)
            w = np.abs(w) / np.linalg.norm(w, ord=1)

            ttrw = np.array([0, 0])
            terminal = False
            env.reset()
            cnt = 0

            hq, Q, _ = agent.predict(torch.from_numpy(w).type(FloatTensor))


            while not terminal:
                state = env.observe()
                action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
                next_state, reward, terminal = env.step(action)
                if cnt > 50:
                    terminal = True
                ttrw = ttrw + reward * np.power(args.gamma, cnt)
                cnt += 1

            act_x.append(ttrw[0])
            act_y.append(ttrw[1])


        act = np.vstack((act_x,act_y))
        act = act.transpose()
        obj = np.vstack((dis_treasure,dis_time))
        obj = obj.transpose()
        act_precition = find_in(act, obj, 2)
        act_recall = find_in(obj, act, 2)
        act_f1 = 2 * act_precition * act_recall / (act_precition + act_recall)
        pred_f1 = 0.0
        policy_f1.append(act_f1)
        print('mean f1:', np.mean(policy_f1))
        print('std dev f1:', np.var(policy_f1)**0.5)
    
    print('mean f1:', np.mean(policy_f1))
    print('std dev f1:', np.var(policy_f1)**0.5)
    
    # Create and style traces(())
    trace_pareto = dict(x=dis_treasure,
                        y=dis_time,
                        mode="markers+lines",
                        type='custom',
                        marker=dict(
                            symbol="circle",
                            size=10),
                        line=dict(
                            width=1,
                            dash='dash'),
                        name='Pareto')

    act_pareto = dict(x=act_x,
                      y=act_y,
                      mode="markers",
                      type='custom',
                      marker=dict(
                          symbol="circle",
                          size=10),
                      line=dict(
                          width=1,
                          dash='dash'),
                      name='Recovered')

    pred_pareto = dict(x=pred_x,
                       y=pred_y,
                       mode="markers",
                       type='custom',
                       marker=dict(
                           symbol="circle",
                           size=3),
                       line=dict(
                           width=1,
                           dash='dash'),
                       name='Predicted')

    layout = dict(title="Deep Sea Treasture Pareto Frontier",
                  xaxis=dict(title='Teasure Value',
                             zeroline=False),
                  yaxis=dict(title='Time Penalty',
                             zeroline=False))
    print("F1: policy-{}|prediction-{}".format(act_f1, pred_f1))
    # send to visdom
    if args.method == "crl-naive":
        vis._send({'data': [trace_pareto, act_pareto], 'layout': layout})
    elif args.method == "crl-envelope":
        vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})
    elif args.method == "crl-energy":
        vis._send({'data': [trace_pareto, act_pareto, pred_pareto], 'layout': layout})

################# HEATMAP #################

if args.pltmap:
    see_map = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
         [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
         [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
    )[::-1]

    vis.heatmap(X=see_map,
                opts=dict(
                    title="DST Map",
                    xmin=-10,
                    xmax=16.6))

if args.pltdemo:
    see_map = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
         [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
         [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
         [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
    )[::-1]

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # generate an agent for plotting
    agent = None
    from crl.envelope.meta_dst import MetaAgent
    model = torch.load(model_loc)
    agent = MetaAgent(model, None, args, is_train=False)

    new_episode = True

    while new_episode:

        dy_map = np.copy(see_map)
        dy_map[10 - 0, 0] = -3

        win = vis.heatmap(X=dy_map,
                          opts=dict(
                              title="DST Map",
                              xmin=-10,
                              xmax=16.6))

        w1 = float(input("treasure weight: "))
        w2 = float(input("time weight: "))
        w = np.array([w1, w2])
        w = np.abs(w) / np.linalg.norm(w, ord=1)
        # w = np.random.dirichlet(np.ones(2))
        ttrw = np.array([0, 0])
        terminal = False
        env.reset()
        cnt = 0
        while not terminal:
            state = env.observe()
            action = agent.act(state, preference=torch.from_numpy(w).type(FloatTensor))
            next_state, reward, terminal = env.step(action)
            dy_map[10 - next_state[0], next_state[1]] = -3
            vis.heatmap(X=dy_map,
                        win=win,
                        opts=dict(
                            title="DST Map",
                            xmin=-10,
                            xmax=14.5))
            Timer.sleep(.5)
            if cnt > 50:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1
        print("final reward: treasure %0.2f, time %0.2f, tot %0.2f" % (ttrw[0], ttrw[1], w.dot(ttrw)))
        new_episode = int(input("try again? 1: Yes | 0: No "))
