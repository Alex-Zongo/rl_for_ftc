import numpy as np


class GaussianNoise:
    """A simple gaussian noise generator:
    """

    def __init__(self, action_dim, sigma=0.2, mu=0):
        self.action_dim = action_dim
        self.sigma = sigma
        self.mu = mu

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)


class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhnlenbeck process based on stackexchange question:
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(self.action_dim)
        self.X = self.X + dx
        return self.X
