import torch
import numpy as np
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from .models.simple_model import MLP

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Exemplar(nn.Module):
    def __init__(self, state_dim, hid_dim, learning_rate, kl_weight, device, n_layers):
        super().__init__()
        
        self.state_dim = state_dim
        self.hid_dim = hid_dim  #Hidden Layer num nodes
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.device = device

        self.clip_grad_norm = 1
        
        n_layers = n_layers

        self.encoder1 = self.make_encoder(state_dim, int(self.hid_dim/2), n_layers=n_layers, hid_size=self.hid_dim)
        self.encoder2 = self.make_encoder(state_dim, int(self.hid_dim/2), n_layers=n_layers, hid_size=self.hid_dim)
        
        self.prior_means = torch.zeros(int(self.hid_dim/2)).to(self.device)
        self.prior_cov = torch.eye(int(self.hid_dim/2)).to(self.device)
        
        self.prior = torch.distributions.MultivariateNormal(self.prior_means, self.prior_cov)
        
        self.discriminator = self.make_discriminator(int(self.hid_dim/2)*2, 1, n_layers=n_layers, hid_size=self.hid_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def make_encoder(self, input_size, z_size, n_layers, hid_size):
        return MLP(input_size, z_size, n_layers, hid_size, device=self.device, with_std_dev=True)
    
    def make_discriminator(self, input_size, z_size, n_layers, hid_size):
        return MLP(input_size, z_size, n_layers, hid_size, device=self.device, with_std_dev=False)
    
    def forward(self, state1, state2):
        encoded1_mean, encoded1_std = self.encoder1(torch.Tensor(state1).to(self.device))
        encoded2_mean, encoded2_std = self.encoder2(torch.Tensor(state2).to(self.device))
        
        epsilon1 = self.prior.sample().to(self.device)
        epsilon2 = self.prior.sample().to(self.device)
        
        latent1 = encoded1_mean + (encoded1_std * epsilon1)
        latent2 = encoded2_mean + (encoded2_std * epsilon2)
        
        logit = self.discriminator(torch.cat([latent1, latent2], axis=1)).squeeze()
        
        return logit
    
    def update(self, state1, state2, target):
        log_likelihood = self.get_log_likelihood(state1, state2, target)
        
        encoded1_mean, encoded1_std = self.encoder1(torch.Tensor(state1).to(self.device))
        encoded2_mean, encoded2_std = self.encoder2(torch.Tensor(state2).to(self.device))
        
        encoded1_dist = torch.distributions.MultivariateNormal(encoded1_mean, torch.diag(encoded1_std**2))
        encoded2_dist = torch.distributions.MultivariateNormal(encoded2_mean, torch.diag(encoded2_std**2))
        
        kl1 = torch.distributions.kl.kl_divergence(encoded1_dist, self.prior)
        kl2 = torch.distributions.kl.kl_divergence(encoded2_dist, self.prior)
        
        # print(kl1)
        # print(kl2)

        elbo_loss = -1*(log_likelihood - self.kl_weight * (kl1 + kl2)).mean()
        
        self.optimizer.zero_grad()
        
        elbo_loss.backward()
        
        self.optimizer.step()
        
        return elbo_loss.cpu().detach().numpy(), kl1.cpu().detach().numpy(), kl2.cpu().detach().numpy()

    def get_log_likelihood(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)
                target: np array (batch_size, 1)

            TODO:
                train the density model and return
                    ll: log_likelihood
                    kl: kl divergence
                    elbo: elbo
        """
        logit = self(state1, state2)

        # print(logit)
        discriminator_dist = torch.distributions.Bernoulli(logits=logit)
        log_likelihood = discriminator_dist.log_prob(torch.Tensor(target).to(self.device).squeeze())

        return log_likelihood

    def get_prob(self, state):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE
        
            args:
                state: np array (batch_size, ob_dim)

            TODO:
                likelihood: 
                    evaluate the discriminator D(x,x) on the same input
                prob:
                    compute the probability density of x from the discriminator
                    likelihood (see homework doc)
        """
        likelihood = torch.exp(self.get_log_likelihood(state, state, torch.ones((state.shape[0],1)))).cpu().detach().numpy()
        likelihood = np.squeeze(likelihood)
        return FloatTensor(torch.tensor(likelihood))