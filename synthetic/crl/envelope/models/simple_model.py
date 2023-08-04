import torch
import torch.nn as nn
from torch.nn import init

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class MLP(torch.nn.Module):

    def __init__(self, input_size, output_size, n_layers, num_neurons, device, activation=torch.relu, output_activation=None, with_std_dev=True):
        super(MLP, self).__init__()

        self.n_layers = n_layers
        self.activation = activation
        self.output_activation = output_activation

        input_size = int(input_size)
        output_size = int(output_size)
        n_layers = int(n_layers) 
        num_neurons = int(num_neurons)
        
        self.input_layer = nn.Linear(input_size, num_neurons)
        self.middle_layers = nn.ModuleList([nn.Linear(num_neurons, num_neurons) for i in range(n_layers)])
        self.output_layer = nn.Linear(num_neurons, output_size)
        self.std_dev = None
        
        if with_std_dev:
            self.std_dev = nn.Parameter(torch.zeros(output_size))
        
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.xavier_uniform_(p.weight)
        
        self.to(device)
        

    def forward(self, state):
        s = self.activation(self.input_layer(state))
        for i, layer in enumerate(self.middle_layers):
            s = self.activation(layer(s))

        if self.output_activation is not None:
            return self.output_activation(self.output_layer(s))
        
        if self.std_dev is None:
            return self.output_layer(s)
        
        return self.output_layer(s), self.std_dev.exp()