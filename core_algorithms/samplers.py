import numpy as np
import operator
from scipy.stats import multivariate_normal
from scipy.stats import norm
# from scipy.stats import logsumexp


class BasicSampler:
    """Simple sampler relying on the ask method of optimizers:
    """

    def __init__(self, sample_archive, thetas_archive, **kwargs):
        self.sample_archive = sample_archive
        self.thetas_archive = thetas_archive
        return

    def ask(self, pop_size, optimizer):
        return optimizer.ask(pop_size)


class IMSampler:
    """Importance Mixing sampler optimized for diagonal diagonal cov matrix:
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.old_mu = None
        self.old_cov = None
        self.first = True
        self.old_elite = None

    def ask(self, pop_size, old_samples):
        if self.first:
            mu, cov = self.optimizer.get_distribution_params()
            self.old_mu = mu
            self.old_cov = cov
            self.first = False
            return self.optimizer.ask(pop_size), 0, []

        n_reused = 0
        n_sampled = 0
        mu, cov = self.optimizer.get_distribution_params()

        idx_reused = []
        params = np.zeros((pop_size, self.optimizer.num_params))

        # current pdf:
        def new_log_pdf(z):
            return norm.logpdf(z, loc=mu, scale=np.sqrt(cov)).sum()

        # old pdf:
        def old_log_pdf(z):
            return norm.logpdf(z, loc=self.old_mu, scale=np.sqrt(self.old_cov)).sum()

        # iterating over pops:
        for i in range(self.optimizer.pop_size):
            sample = old_samples[i]  # old candidates:

            if n_reused + n_sampled < pop_size:
                u = np.random.uniform(0, 1)

                # rejection sampling:
                if np.log(u) < new_log_pdf(sample) - old_log_pdf(sample):
                    params[n_reused] = sample
                    idx_reused.append(i)
                    n_reused += 1

            if n_reused + n_sampled < pop_size:
                sample = self.optimizer.ask(1).reshape(-1)
                u = np.random.uniform(0, 1)

                # rejection sampling:
                if np.log(1-u) >= old_log_pdf(sample) - new_log_pdf(sample):
                    params[-n_sampled-1] = sample
                    n_sampled += 1

            if n_reused + n_sampled >= pop_size:
                break

        # filling the rest of the population with new samples:
        cpt = n_reused + n_sampled
        while cpt < pop_size:
            sample = self.optimizer.ask(1).reshape(-1)
            params[cpt-n_sampled] = sample
            cpt += 1

        self.old_mu = mu
        self.old_cov = cov

        return params, n_reused, idx_reused
