from __future__ import absolute_import, division, print_function
import argparse
from datetime import datetime
import imp
import numpy as np
import torch
from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv
# from gym_env_moll.multiobjective import LunarLander
# import gym
import json


parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='ft', metavar='ENVNAME',
                    help='environment to train on: dst | ft | ft5 | ft7')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=246, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.8, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=True, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')

use_cuda =  torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def generate_next_preference(preference, alpha=10000):
    preference = np.array(preference)
    preference += 1e-6
    
    return FloatTensor(np.random.dirichlet(alpha*preference))

def init_log_file(log_file_str):
    with open(log_file_str, mode='w+') as log_file:
        log_file.write('[\n')

def write_log(log_file_str, data, is_json = False):
    with open(log_file_str, mode='a+') as log_file:
        if is_json:
            json.dump(data, log_file)
        else:
            log_file.write(data)


def train(env, agent, args):
    log_file_str = './logs/multihead_Q_no_transition_log_' + args.env_name + '_' + str(datetime.today().strftime("%Y_%m_%d")) + '.json'
    save_loc = './saved_models/'
    save_file_name = 'multihead_Q_no_transition_log_' + args.env_name + '_' + str(datetime.today().strftime("%Y_%m_%d"))

    init_log_file(log_file_str)

    env.reset()
    alpha = args.alpha
    max_steps_in_env = 100
    
    probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])

    for num_eps in range(args.episode_num+1):
        terminal = False
        env.reset()
        loss = 0
        cnt = 0
        tot_reward = 0

        # probe = generate_next_preference(np.random.uniform(size=len(env.reward_spec))/len(env.reward_spec), alpha = 1)
        # probe = torch.randn(len(env.reward_spec))
        # probe = (torch.abs(probe) / torch.norm(probe, p=1)).type(FloatTensor)
        # probe = torch.randn(len(env.reward_spec))
        # probe = (torch.abs(probe)/torch.norm(probe, p=1)).type(FloatTensor)
        # probe = generate_next_preference(torch.randn(len(env.reward_spec)), alpha = 500)
        # if num_eps % 100 == 0:
        #     probe = FloatTensor([0.98, 0.02])
        #     probe = generate_next_preference(probe, 200)
        
        write_log(log_file_str, '[')

        while not terminal:
            state = env.observe()
            action = agent.act(state)
            next_state, reward, terminal = env.step(action)
            
            agent.memorize(state, action, next_state, reward, terminal, probe, probe)
            loss += agent.learn()
            
            if cnt > max_steps_in_env:
                terminal = True
                agent.reset()
            
            tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            cnt = cnt + 1

            if args.log and (num_eps % 50) == 0:
                _, Q, q = agent.predict(probe, state)

                log = {
                    'state':state.tolist(),
                    'action':action,
                    'reward':reward.tolist(),
                    'terminal':terminal,
                    'probe':probe.detach().numpy().tolist(),
                    'q_val': q.tolist(),
                    'cnt': cnt,
                    'num_eps': num_eps
                }

                print('probe', probe.detach().numpy().tolist())
                print('state', log['state'])
                print('action', log['action'])
                print('reward', log['reward'])
                print('q_val', log['q_val'])
                print('Q_val', Q.detach().numpy().tolist())
                print('tot_reward', tot_reward)
                print('cnt', log['cnt'])
                print('num_eps', log['num_eps'])
                print('eps', agent.epsilon)
                print('---------------------------------------')

                write_log(log_file_str, log, True)

                if not terminal:
                    write_log(log_file_str, ',\n')
                else:
                    write_log(log_file_str, '\n],\n')


        _, Q, q = agent.predict(probe)

        if args.env_name == "dst":
            act_1 = q[0, 3]
            act_2 = q[0, 1]
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            act_1 = q[0, 1]
            act_2 = q[0, 0]

        if args.method == "crl-naive":
            act_1 = act_1.data.cpu()
            act_2 = act_2.data.cpu()
        elif args.method == "crl-envelope":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        elif args.method == "crl-energy":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        print("end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f; the probe is %0.2f | %0.2f; loss: %0.4f" % (
            num_eps,
            tot_reward,
            act_1,
            act_2,
            probe[0],
            probe[1],
            # q__max,
            loss / cnt))


        if (num_eps+1) % 500 == 0:
            agent.save(save_loc, save_file_name+"_eps_"+str(num_eps))

    
    agent.save(save_loc, save_file_name+"_eps_"+str(num_eps))


if __name__ == '__main__':
    args = parser.parse_args()

    # setup the environment
    # args.env_name = 'Lunar'
    # env = gym.make('gym.envs.multiobjective/LunarLander')
    env = MultiObjectiveEnv(args.env_name)
    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

    # generate an agent for initial training
    agent = None
    
    args.alpha = 4000

    from crl.envelope.meta_no_transition import MetaAgent
    from crl.envelope.models.multiheadoutput import EnvelopeLinearCQN

    if args.serialize:
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))

    model = EnvelopeLinearCQN(state_size, action_size, reward_size)    
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
