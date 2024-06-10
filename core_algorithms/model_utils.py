"""================================= Utils for the learning module ================"""
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# +++++++++++++++++++++++++++++++++++++++++++++++
# Non-linear activation functions:
# +++++++++++++++++++++++++++++++++++++++++++++++

activations = {
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),  # or nn.ReLU()
}


def soft_update(target, source, tau):
    """
    Soft update of the target network parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)


def hard_update(target, source):
    """
    Hard update of the target network parameters.
    θ_target = θ_local
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# Noise


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class GaussianNoise:

    def __init__(self, action_dimension, std=0.1, mu=0):
        self.action_dimension = action_dimension
        self.sd = std
        self.mu = mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        return np.random.normal(self.mu, self.sd, self.action_dimension)


class OUNoise:
    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# +++++++++++++++++++++++++++++++++++++++++
        # Convertors:
# +++++++++++++++++++++++++++++++++++++++++


def to_numpy(var):
    """
    Convert a pytorch tensor to a numpy array.
    """
    return var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad)


def pickle_obj(filename, object):
    handle = open(filename, 'wb')
    pickle.dump(object, handle)


def unpickle_obj(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def odict_to_numpy(odict):
    l = list(odict.values())
    state = l[0]
    for i in range(i, len(l)):
        if isinstance(l[i], np.ndarray):
            state = np.concatenate((state, l[i]))
        else:
            state = np.concatenate((state, np.array(l[i])))
    return state


def min_max_normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    return (x - min_x) / (max_x - min_x)


def is_lnorm_key(key):
    return key.startswith('lnorm')


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    # v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)
