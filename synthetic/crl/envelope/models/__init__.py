from __future__ import absolute_import, division, print_function
from .linear import EnvelopeLinearCQN
from .multiheadoutput import EnvelopeLinearCQN
from .simple_model import MLP

def get_new_model(name, state_size, action_size, reward_size):
    if name == 'linear':
        return EnvelopeLinearCQN(state_size, action_size, reward_size)
    else:
        print("model %s doesn't exist." % (name))
        return None
