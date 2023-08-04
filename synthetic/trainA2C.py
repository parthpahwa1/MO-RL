from __future__ import absolute_import, division, print_function
import argparse
from email import policy
from os import stat
from statistics import mode
import numpy as np
import json
import torch
from utils.monitor import Monitor
from envs.mo_env import MultiObjectiveEnv
from crl.a2c.models import A2C
from torch.autograd import Variable
from datetime import datetime

parser = argparse.ArgumentParser(description='MORL')

# CONFIG
parser.add_argument('--env-name', default='ft', metavar='ENVNAME',
                    help='environment to train on: dst | ft | ft5 | ft7')
parser.add_argument('--method', default='crl-a2c', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='wLSE', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | R   prop')
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

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def generate_next_preference(preference, alpha=1):
    preference = np.random.uniform(size=len(preference))/2

    preference += 1e-6
    
    return FloatTensor(np.random.dirichlet(preference))

def init_log_file(log_file_str):
    with open(log_file_str, mode='w+') as log_file:
        log_file.write('[\n')

def write_log(log_file_str, data, is_json = False):
    with open(log_file_str, mode='a+') as log_file:
        if is_json:
            json.dumps(data)
        else:
            log_file.write(data)


def make_episode_data(curr_state_batch, curr_preference_batch, action_batch, next_state_batch, next_preference_batch, reward_batch_scalar, reward_batch_vector, terminal_batch, curr_value_scalar_batch, curr_value_vector_batch, next_value_batch, next_value_vector_batch, gamma):
    episode_data = {}
    
    discounted_return_scalar = np.empty(shape=curr_value_scalar_batch.shape)
    discounted_return_vector = np.zeros(shape=reward_batch_vector.shape)

    adv = np.empty([curr_value_scalar_batch.shape[0]])
    adv_vector = np.empty([curr_value_scalar_batch.shape[0]])
    
    running_add = next_value_batch[-1]
    running_add_vector = next_value_vector_batch[-1]
    
    for i in range(len(action_batch)-1, -1, -1):
        running_add = reward_batch_scalar[i] + gamma*running_add*(1-terminal_batch[i])
        discounted_return_scalar[i] = running_add

        running_add_vector = reward_batch_vector[i] + gamma*running_add_vector*(1-terminal_batch[i])
        discounted_return_vector[i] = running_add_vector

    adv = discounted_return_scalar - curr_value_scalar_batch
    adv_vector = discounted_return_vector - curr_value_vector_batch

    episode_data['target_scalar'] = discounted_return_scalar.copy()
    episode_data['target_vector'] = discounted_return_vector.copy()
    episode_data['curr_state'] = curr_state_batch
    episode_data['curr_preference'] = curr_preference_batch
    episode_data['next_state'] = next_state_batch
    episode_data['next_preference'] = next_preference_batch
    episode_data['action'] = action_batch.copy()
    episode_data['advantage'] = adv.copy()
    episode_data['advantage_vector'] = adv_vector.copy()

    return episode_data


def train(env, agent, args):
    log_file_str = './logs/multihead_A2C_log_' + args.env_name + '_' + str(datetime.today().strftime("%Y_%m_%d")) + '.json'
    save_loc = './saved_models/'
    save_file_name = 'multihead_A2C_log_' + args.env_name + '_' + str(datetime.today().strftime("%Y_%m_%d"))
    
    init_log_file(log_file_str)
    env.reset()
    fixed_start_state = env.observe()
    env.reset()
    
    alpha = 3000
    
    args.episode_num = 500000

    probe = torch.randn(len(env.reward_spec))
    probe = (torch.abs(probe)/torch.norm(probe, p=1)).type(FloatTensor)
    probe = generate_next_preference(probe)

    reward_sum = 0
    time_penalty_sum = 0
    
    policy_dict = {
    }
    
    print('Started')
    for num_eps in range(args.episode_num+1):
        terminal = False
        env.reset()
        cnt = 0
        
        state_batch, action_batch, next_state_batch, reward_scalar_batch, reward_vector_batch, terminal_batch = [], [], [], [], [], []
        preferece_batch, next_preference_batch = [], []

        probe = torch.randn(len(env.reward_spec))
        probe = (torch.abs(probe)/torch.norm(probe, p=1)).type(FloatTensor)
        probe = generate_next_preference(probe)
        
        
        while not terminal:
            state = env.observe()
            action = agent.get_action(state, probe)
            
            next_state, reward, terminal = env.step(action)
            next_preference = probe

            if cnt > 40 or terminal:
                agent.reset()
                terminal = True
            

            state_batch.append(state.copy())
            action_batch.append(action.copy())
            next_state_batch.append(next_state.copy())
            reward_scalar_batch.append(np.dot(probe, reward.copy()))
            reward_vector_batch.append(reward.copy())
            terminal_batch.append(terminal)
            preferece_batch.append(probe)
            next_preference_batch.append(next_preference)

            if (num_eps + 1) % 1000 == 0 or reward[0] > 8 or probe[0] > 0.79:
                print('state', state,'next_state', next_state)
                print('action', action)
                print('preference', probe,'next_preference', next_preference)
                print('reward', reward)
                print('cnt', cnt)
                print('eps', num_eps)
                print('--------------------')
            
            probe = next_preference
            cnt = cnt + 1

        state_batch_stack = [np.stack(state_batch.copy())]
        action_batch_stack = [np.stack(action_batch.copy())]
        next_state_batch_stack = [np.stack(next_state_batch.copy())]
        reward_scalar_batch_stack = [np.stack(reward_scalar_batch.copy())]
        reward_vector_batch_stack = [np.stack(reward_vector_batch.copy())]
        terminal_batch_stack = [np.stack(terminal_batch.copy())]
        preferece_batch_stack = [np.stack(preferece_batch.copy())]
        next_preference_batch_stack = [np.stack(next_preference_batch.copy())]
        
        for i in range(10):
            probe = torch.randn(len(env.reward_spec))
            probe = (torch.abs(probe)/torch.norm(probe, p=1)).type(FloatTensor)
            probe = generate_next_preference(probe)
            probe = [probe]
            
            state_batch_stack.append(np.stack(state_batch))
            action_batch_stack.append(np.stack(action_batch))
            next_state_batch_stack.append(np.stack(next_state_batch))
            reward_scalar_batch_stack.append(np.stack(reward_scalar_batch))
            reward_vector_batch_stack.append(np.stack(reward_vector_batch))
            terminal_batch_stack.append(np.stack(terminal_batch))
            preferece_batch_stack.append(np.stack(probe*len(state_batch)))
            next_preference_batch_stack.append(np.stack(probe*len(state_batch)))

        for i in range(len(state_batch_stack)):
            state_batch = state_batch_stack[i]
            action_batch = action_batch_stack[i]
            next_state_batch = next_state_batch_stack[i]
            reward_scalar_batch = reward_scalar_batch_stack[i]
            reward_vector_batch = reward_vector_batch_stack[i]
            terminal_batch = terminal_batch_stack[i]
            preferece_batch = preferece_batch_stack[i]
            next_preference_batch = next_preference_batch_stack[i]
            # print(i)
            # print(state_batch_stack)  
            curr_value, next_value, _ = agent.forward_transition(state_batch, next_state_batch, preferece_batch, next_preference_batch)
            episode_data = {}

            if state_batch.shape[0] == 1:
                curr_value_scalar = curr_value[0]
                curr_value_vector = np.reshape(curr_value[1], (1, curr_value[1].shape[0]))

                next_value_scalar = next_value[0]
                next_value_vector = np.reshape(next_value[1], (1, next_value[1].shape[0]))

                adv = reward_scalar_batch - curr_value_scalar
                adv_vector = reward_vector_batch - curr_value_vector
                
                episode_data['curr_state'] = state_batch
                episode_data['next_state'] = next_state_batch
                episode_data['target_scalar'] = reward_scalar_batch
                episode_data['action'] = action_batch
                episode_data['advantage'] = adv
                episode_data['advantage_vector'] = adv_vector
                episode_data['target_vector'] = reward_vector_batch
                episode_data['curr_preference'] = preferece_batch
                episode_data['next_preference'] = next_preference_batch


            else:
                episode_data = make_episode_data(state_batch, preferece_batch, action_batch, next_state_batch, next_preference_batch,  
                    reward_scalar_batch, reward_vector_batch, terminal_batch, curr_value[0], 
                    curr_value[1], next_value[0], next_value[1], args.gamma)
            
            reward_sum = episode_data['target_scalar'][0]
            
            agent.learn(episode_data)

        # if (num_eps + 1) % 99 == 0:
        #     print(episode_data)
        
        preference = torch.randn(len(env.reward_spec))
        preference = (torch.abs(preference)/torch.norm(preference, p=1)).type(FloatTensor)
        
        value_for_start_scalar, value_for_start_vector, policy = agent.get_value_for_state(fixed_start_state, preference)
        value_for_start_scalar = value_for_start_scalar.detach().numpy()
        value_for_start_vector = value_for_start_vector.detach().numpy()
        policy = policy.detach().numpy()

        if (num_eps + 1) % 50000 == 0:

            print('--------------------------------------')
            print('value_for_start', value_for_start_scalar, value_for_start_vector, 'policy', policy)
            print('average_reward', reward_sum)
            print('probe', preference)
            print('num_eps', num_eps)
            print('--------------------------------------')
            reward_sum = 0
            agent.save(save_loc, save_file_name+'_num_eps_'+str(num_eps))


if __name__ == '__main__':
    args = parser.parse_args()

    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # get state / action / reward sizes
    state_size = len(env.state_spec)
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    reward_size = len(env.reward_spec)

    parameter_size = 1

    # generate an agent for initial training
    agent = None
    
    from crl.a2c.meta import MetaAgent
    from crl.a2c.models.A2C import A2C
    
    model = A2C(state_size, action_size, reward_size, parameter_size)
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
