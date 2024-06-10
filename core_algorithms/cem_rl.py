import random
import gymnasium as gym
# from core_algorithms.utils import calc_smoothness
from environments.config import select_env
from copy import copy, deepcopy
import torch
import numpy as np
# from ppo_continous_actions import Agent
from parameters_es import ESParameters

# from torch.func import stack_module_state, functional_call
# from torch import vmap


class CEM:
    '''Cross Entropy Methods Algorithm '''

    def __init__(self, num_params, params: ESParameters, sigma_init=0.1, mu_init=None, pop_size=10, sigma_decay=0.999, sigma_limit=0.01, damp=1e-3, damp_limit=1e-5, parents=None, elitism=False, antithetic=False, adaptation=False):
        """"""
        self.seed = np.random.seed(params.seed)
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
        # self.multi_envs = gym.vector.AsyncVectorEnv(
        #     [make_env() for _ in range(pop_size)])
        # self.single_env = gym.vector.AsyncVectorEnv(
        #     [make_env() for _ in range(1)])
        # self.ag_multi = Agent(envs=self.multi_envs)
        self.params = params
        self.num_params = num_params

        self.mu = np.zeros(
            self.num_params) if mu_init is None else np.array(mu_init)
        self.sigma = params.sigma_init if params.sigma_init else sigma_init
        self.sigma_decay = params.sigma_decay if params.sigma_decay else sigma_decay
        self.sigma_limit = params.sigma_limit if params.sigma_limit else sigma_limit
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)
        self.best_cov = deepcopy(self.cov)

        # elite stuff:
        self.elitism = elitism
        # self.elite = np.sqrt(self.sigma) * np.random.randn(self.num_params)
        self.elite = deepcopy(self.mu) if mu_init is not None else np.sqrt(
            self.sigma) * np.random.randn(self.num_params)
        self.elite_score = -np.inf
        self.best_elite_so_far = deepcopy(self.elite)
        self.best_elite_so_far_score = -np.inf
        self.best_mu_so_far = deepcopy(self.mu)
        self.best_mu_so_far_score = -np.inf
        self.mu_score = -np.inf

        self.rl_agent = deepcopy(self.mu)
        self.rl_agent_score = -np.inf
        # sampling stuff:
        self.pop_size = pop_size
        self.antithetic = antithetic

        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"

        # sigma and cov adaptation stuff:
        self.adaptation = adaptation
        if parents is None or parents <= 0:
            self.parents = pop_size//2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents+1)/i)
                                for i in range(1, self.parents+1)])
        self.weights /= self.weights.sum()
        self.first_interaction = True

    def ask(self, pop_size):
        """ Ask for candidate solutions"""
        # np.random.seed(self.params.seed)
        if self.antithetic and not pop_size % 2:
            print('Antithetic sampling')
            epsilon_half = np.random.randn(pop_size//2, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.first_interaction:
            inds[-1] = self.mu
            self.first_interaction = False
        if self.elitism and not self.first_interaction:
            inds[0] = self.best_mu_so_far
            inds[-1] = self.best_elite_so_far
            if self.rl_agent_score > 1.05 * self.mu_score:
                inds[-2] = self.rl_agent
            # inds[-3] = self.best_elite_so_far
        return inds

    # def pop_agent_stack(self, pop):

    #     ag_stack = []
    #     if len(pop.shape) == 1:
    #         self.ag_multi.inject_parameters(pop)
    #         ag_stack.append(self.ag_multi.actor_mean)
    #     else:
    #         for cand in pop:
    #             self.ag_multi.inject_parameters(cand)
    #             ag_stack.append(self.ag_multi.actor_mean)
    #     return ag_stack

    # def evaluate_pop(self, pop):
    #     N = pop.shape[0] if len(pop.shape) > 1 else 1
    #     fitness_table = np.zeros(N)
    #     actions_sm_data = np.zeros(
    #         (self.params.max_timesteps_per_episode, N, 3))
    #     ag_stack = self.pop_agent_stack(pop)
    #     params, buffers = stack_module_state(ag_stack)
    #     base_model = deepcopy(ag_stack[0])
    #     base_model = base_model.to('meta')

    #     def fmodel(params, buffers, x):
    #         return functional_call(base_model, (params, buffers), (x,))

    #     obs = self.multi_envs.reset(
    #     )[0] if N > 1 else self.single_env.reset()[0]
    #     for step in range(self.params.max_timesteps_per_episode):
    #         actions = vmap(fmodel)(params, buffers,
    #                                torch.FloatTensor(obs)).detach().numpy()
    #         obs, reward, done, truncated, info = self.multi_envs.step(
    #             actions) if N > 1 else self.single_env.step(actions)
    #         actions_sm_data[step] = actions
    #         fitness_table += reward

    #         if all(done) or all(truncated):
    #             break
    #     sms = calc_smoothness(actions_sm_data)
    #     assert len(fitness_table) == len(sms)
    #     fitness_table += sms
    #     return fitness_table

    # def step(self, model, params, buffers, obs):
    #     actions = vmap(model)(params, buffers, torch.FloatTensor(obs).detach().numpy())
    #     next_obs, reward, done, truncated, info = self.multi_envs.step(actions.cpu().numpy())

    def tell(self, solutions, fitness_table):
        """ Updates the distribution parameters of the candidate solutions and scores table"""
        scores = np.array(fitness_table)
        # times -1 so that maximization becomes minimization problem:

        idx_sorted = np.argsort(scores)[::-1]

        old_mu = deepcopy(self.mu)
        self.damp = self.damp * self.tau + (1-self.tau) * self.damp_limit

        if self.params.fixed_softmax_cem:
            print('>> update with softmax fitness values')
            elite_diff = (scores[idx_sorted[:self.parents]] + 10000)/1000
            p = 1 / (1+np.exp(-elite_diff))
            self.mu = (self.weights * p) @ solutions[idx_sorted[:self.parents]]
        else:
            self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        z = (solutions[idx_sorted[:self.parents]] - old_mu)

        # *** sigma adaptation and cov:
        if self.adaptation:
            # ====== version 1
            if self.sigma > self.sigma_limit:
                self.sigma *= self.sigma_decay
            # ====== version 2
            # if scores[idx_sorted[0]] > 1.05 * self.elite_score:
            #     self.sigma *= 0.95
            # else:
            #     self.sigma *= 1.05

            # ===== Covariance
            self.cov = self.weights @ (z * z)
            self.cov = self.sigma * self.cov / \
                np.linalg.norm(self.cov)

        else:
            # ===== Covariance
            self.cov = 1 / self.parents * \
                self.weights @ (z * z) + self.damp * np.ones(self.num_params)

        # ====== Elite
        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        if self.elite_score > self.best_elite_so_far_score:
            self.best_elite_so_far = deepcopy(self.elite)
            self.best_elite_so_far_score = copy(self.elite_score)
        # === best mu:
        self.mu_score = self.evaluate_pop(self.mu)[0]
        if self.mu_score > self.best_mu_so_far_score:
            self.best_mu_so_far = deepcopy(self.mu)
            self.best_mu_so_far_score = copy(self.mu_score)

    def get_distribution_params(self):
        """ Returns the distribution parameters: mu and sigma"""
        return self.mu, self.sigma

    def save_model(self, parameters):
        pass


def make_env():
    def thunk():
        seed = 1
        env_name = "PHlab_attitude_nominal"
        env = select_env(env_name, render_mode=False,
                         realtime=False, use_state_history=False,
                         conform_with_sb=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(
        #     env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(
        #     env, lambda reward: np.clip(reward, -10, 10))
        # env.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


class CEMA:
    """ Cross-Entropy Methods with sigma adaptation"""

    def __init__(self, num_params, params, sigma_init=1e-3, mu_init=None, pop_size=10, damp=1e-3, damp_limit=1e-5, parents=None, elitism=False, antithetic=False):
        self.params = params
        self.num_params = num_params

        self.mu = np.zeros(
            self.num_params) if mu_init is None else np.array(mu_init)
        self.sigma = params.sigma_init if params.sigma_init else sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff:
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = -np.inf

        # sampling stuff:
        self.pop_size = pop_size
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size//2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents+1)/i)
                                for i in range(1, self.parents+1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """ Ask for candidate solutions"""
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size//2, self.num_params)
            epsilon = np.concatenate([epsilon_half, -epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite
        return inds

    def tell(self, solutions, fitness_table):
        """ Updates the distribution parameters of the candidate solutions and scores table"""
        scores = np.array(fitness_table)
        # times -1 so that maximization becomes minimization problem:
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1-self.tau) * self.damp_limit
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # sigma adaptation:
        if scores[idx_sorted[0]] > 0.95 * self.elite_score:
            self.sigma *= 0.95
        else:
            self.sigma *= 1.05

        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = self.weights @ (z * z)
        self.cov = self.sigma * self.cov / np.linalg.norm(self.cov)

        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]

    def get_distribution_params(self):
        """ Returns the distribution parameters: mu and sigma"""
        return self.mu, self.sigma
