from copy import deepcopy
import torch.nn.functional as F
from core_algorithms.utils import Episode
import os
from tqdm import tqdm
from core_algorithms.utils import calc_smoothness
from core_algorithms.some_actor_model import Actor, RNN_Actor
import numpy as np
import torch
# from parameters import Parameters
from parameters_es import ESParameters


def compute_weight_decay(weight_decay, model_params):
    model_params_grid = torch.tensor(model_params)
    return -weight_decay * torch.mean(model_params_grid * model_params_grid, dim=1)


def evaluate(env, params: ESParameters, agent: Actor or RNN_Actor, is_action_noise: bool, store_transition: bool):
    """Evaluate the agent, one episode is played and the total reward is returned.

    Args:
        env (gym or gym wrapped): _description_
        agent (nn): the actor
        is_action_noise (bool): add noise to the action
        store_transition (bool): store the transition in replay buffer
        use_caps (bool): use CAPS or not.
    Returns:
        total reward or fitness
        smoothness
    """
    # if self.args.use_caps:
    #     self.caps_dict = {
    #         'lambda_s': 0.5,
    #         'lambda_t': 0.1,
    #         'eps_sd': 0.05,
    #     }
    if params.use_caps:
        caps_params = {
            "eps_sd": 0.05,
            "lambda_t": 0.1,
            "lambda_s": 0.5
        }

    state_lst, rewards, action_lst, caps_losses = [], [], [], []
    done = False
    obs, _ = env.reset(seed=0)
    agent.eval()
    while not done:
        action = agent.select_action(obs)
        if is_action_noise:
            action = np.clip(
                action + np.random.normal(
                    0, 0.1, size=env.action_space.shape[0]), -1.0, 1.0)

        if 'lunar' in params.env_name:
            next_obs, reward, done, truncated, _ = env.step(action.flatten())
            action_lst.append(action.flatten())
            state_lst.append(obs)
        else:
            next_obs, reward, done, info = env.step(action.flatten())
            action_lst.append(env.last_u)
            state_lst.append(env.x)

            if params.use_caps:
                obs_bar = obs + \
                    np.random.normal(0, caps_params["eps_sd"], size=obs.shape)
                action_bar = agent.select_action(obs_bar)
                caps_loss = caps_params["lambda_t"] * \
                    F.mse_loss(torch.tensor(action), torch.tensor(action_bar)) + \
                    caps_params["lambda_s"] * \
                    F.mse_loss(torch.tensor(
                        action_lst[-1]), torch.tensor(action_bar))

                caps_losses.append(caps_loss.item())

        rewards.append(reward)
        if store_transition:
            transition = (obs, action, next_obs, reward, float(done))
            agent.buffer.push(*transition)

        obs = next_obs
    env.close()
    if caps_losses:
        caps_losses = np.asarray(caps_losses)
        caps_overall = np.sum(caps_losses)
    else:
        caps_overall = -1.0
    actions = np.array(action_lst)
    smoothness = calc_smoothness(actions, plot_spectra=False)

    fitness = np.sum(rewards) + smoothness
    if params.use_caps:
        fitness = fitness - caps_overall

    if 'lunar' not in params.env_name:
        episode = Episode(
            fitness=fitness,
            smoothness=smoothness,
            caps_loss=caps_overall,
            length=info['t'], state_history=state_lst,
            ref_signals=info['ref'],
            actions=actions,
            reward_lst=rewards,
        )
    else:
        episode = Episode(
            fitness=fitness,
            smoothness=smoothness,
            actions=actions,
            reward_lst=rewards,
            state_history=state_lst,
        )
    return episode


def fitness_function(solutions: torch.tensor, params: ESParameters, env):
    """fitness function

    Args:
        solutions (_type_): _description_
        params (_type_): _description_
        env (_type_): _description_
    Returns:
        fitness: A tensor of shape (pop_size, 1)
        sm_avg: the average smoothness of the population
        sm_sd: the standard deviation of the population's smoothness
    """
    num_evals = 3
    if len(solutions.shape) != 2:
        solutions = solutions.reshape(1, -1)

    lamda, N = solutions.shape
    smoothness_lst = []
    episode_length = []

    caps_losses = []
    fitness_lst = torch.zeros((num_evals, lamda))
    if params.use_rnn:
        agent = RNN_Actor(args=params, init=True)
    else:
        agent = Actor(args=params, init=True)

    for j in tqdm(range(lamda), total=lamda, desc='Population Evaluation', colour='green'):

        for i in range(num_evals):
            agent.inject_parameters(solutions[j])
            episode = evaluate(
                env, params, agent, is_action_noise=True, store_transition=False)
            smoothness_lst.append(episode.smoothness)
            episode_length.append(episode.length)
            caps_losses.append(episode.caps_loss)
            fitness_lst[i, j] = episode.fitness

    pop_fit = fitness_lst.mean(dim=0).reshape(-1, 1)
    smoothness_lst = np.asarray(smoothness_lst)
    caps_losses = np.asarray(caps_losses)
    caps_loss = np.mean(caps_losses)
    sm_avg = np.median(smoothness_lst)
    sm_sd = smoothness_lst.std()
    episode_length = np.asarray(episode_length)
    ep_length = np.mean(episode_length)
    ep_length_sd = np.std(episode_length)
    return pop_fit, sm_avg, sm_sd, ep_length, ep_length_sd, caps_loss, episode


class sepCMAES:
    ''' Implementation of the separable CMA-ES algorithm
    https://en.wikipedia.org/wiki/CMA-ES#Example_code_in_MATLAB/Octave
    '''

    def __init__(self, num_params, params: ESParameters,
                 mu_init=None,
                 sigma_init=1,
                 step_size_init=1,
                 pop_size=10,
                 antithetic=False,
                 weight_decay=0.01,
                 elitism=False,
                 rank_fitness=True):

        # distribution params:
        self.params = params
        np.random.seed(self.params.seed)
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.mu = np.zeros(
            self.num_params) if mu_init is None else np.array(mu_init)

        self.antithetic = antithetic

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma_init) * np.random.rand(self.num_params)
        self.elite_score = -np.inf

        # stuff
        self.step_size = step_size_init
        self.p_c = np.zeros(self.num_params)
        self.p_s = np.zeros(self.num_params)
        self.cov = sigma_init * np.ones(num_params)

        # selection parameters
        self.pop_size = pop_size
        self.parents = pop_size // 2
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()
        self.parents_eff = 1 / (self.weights ** 2).sum()
        self.rank_fitness = rank_fitness
        self.weight_decay = weight_decay

        # adaptation  parameters
        self.g = 1
        self.c_s = (self.parents_eff + 2) / \
            (self.num_params + self.parents_eff + 3)
        self.c_c = 4 / (self.num_params + 4)
        self.c_cov = 1 / self.parents_eff * 2 / ((self.num_params + np.sqrt(2)) ** 2) + (1 - 1 / self.parents_eff) * \
            min(1, (2 * self.parents_eff - 1) /
                (self.parents_eff + (self.num_params + 2) ** 2))
        self.c_cov *= (self.num_params + 2) / 3
        self.d_s = 1 + 2 * \
            max(0, np.sqrt((self.parents_eff - 1) /
                           (self.num_params + 1) - 1)) + self.c_s
        self.chi = np.sqrt(self.num_params) * (1 - 1 / (4 *
                                                        self.num_params) + 1 / (21 * self.num_params ** 2))

    def ask(self, pop_size):
        """Returns a list of candidate solutions

        Args:
            pop_size (_type_): _description_
        """
        if self.antithetic:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)

        solutions = self.mu + self.step_size * epsilon * np.sqrt(self.cov)
        if self.elitism:
            solutions[-1] = self.elite
        return solutions

    def tell(self, solutions, scores):
        """Update the distribution

        Args:
            solutions (_type_): _description_
            scores (_type_): _description_
        """
        scores = np.array(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        # update mean
        old_mu = deepcopy(self.mu)
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]
        z = 1 / self.step_size * 1 / \
            np.sqrt(self.cov) * (solutions[idx_sorted[:self.parents]] - old_mu)
        z_w = self.weights @ z

        # update evolution paths
        self.p_s = (1 - self.c_s) * self.p_s + \
            np.sqrt(self.c_s * (2 - self.c_s) * self.parents_eff) * z_w

        tmp_1 = np.linalg.norm(self.p_s) / np.sqrt(1 - (1 - self.c_s) ** (2 * self.g)) \
            <= self.chi * (1.4 + 2 / (self.num_params + 1))

        self.p_c = (1 - self.c_c) * self.p_c + \
            tmp_1 * np.sqrt(self.c_c * (2 - self.c_c)
                            * self.parents_eff) * np.sqrt(self.cov) * z_w

        # update covariance matrix
        self.cov = (1 - self.c_cov) * self.cov + \
            self.c_cov * 1 / self.parents_eff * self.p_c * self.p_c + \
            self.c_cov * (1 - 1 / self.parents_eff) * \
            (self.weights @ (self.cov * z * z))

        # update step size
        self.step_size *= np.exp((self.c_s / self.d_s) *
                                 (np.linalg.norm(self.p_s) / self.chi - 1))
        self.g += 1

        # print(self.cov)
        self.elite = solutions[idx_sorted[0]]
        self.elite_score = -scores[idx_sorted[0]]
        # return idx_sorted[:self.parents]

    def get_distribution_params(self):
        """distribution params, mean and covariance matrix

        Returns:
            _type_: _description_
        """
        return np.copy(self.mu), np.copy(self.step_size)**2 * np.copy(self.cov)


class CMAES_v2:

    def __init__(self, num_params, params: ESParameters, sigma_init=1e-3, mu_init=None, pop_size=20, elitism=False):
        self.params = params
        self.num_params = num_params
        np.random.seed(self.params.seed)
        self.sigma_init = params.sigma_init if params.sigma_init else sigma_init
        self.pop_size = params.pop_size if self.params.pop_size else pop_size
        self.weight_decay = params.weight_decay if self.params.weight_decay else 0.01
        self.solutions = None
        self.generations = 0
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma_init) * np.random.rand(self.num_params)
        self.elite_score = None
        self.mu = np.zeros(
            self.num_params) if mu_init is None else np.array(mu_init)

        import cma
        self.es = cma.CMAEvolutionStrategy(
            self.num_params * [0], self.sigma_init, {'popsize': self.pop_size})

    def ask(self, pop_size):
        '''Returns a list of solutions'''
        solutions = np.array(self.es.ask(number=pop_size))
        if self.elitism:
            solutions[-1] = self.elite
        return solutions

    def tell(self, solutions, score_table):
        score_table = torch.tensor(np.array(score_table))
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(
                self.weight_decay, solutions)

            score_table += l2_decay
        # convert minimizer to maximizer:
        self.es.tell(solutions,
                     (-score_table).tolist())

        self.mu = self.current_param()
        self.elite = self.best_param()
        self.elite_score = self.result()[1]
        self.sigma = self.result()[3]

    def get_distribution_params(self):
        return self.mu, self.sigma

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):
        '''return the best params so far along with historically best reward, current reward, sigma'''
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])


class CMAES:
    '''CMA-ES algorithm implementation (Wrapper)'''

    def __init__(self, num_params, params: ESParameters, sigma_init=0.10, pop_size=20, weight_decay=0.01, max_generations=500):

        self.params = params
        self.num_params = params.num_params if params.num_params else num_params
        self.max_generations = params.max_generations if params.max_generations else max_generations
        self.sigma_init = params.sigma_init if params.sigma_init else sigma_init
        self.pop_size = params.pop_size if self.params.pop_size else pop_size
        self.weight_decay = weight_decay
        self.solutions = None
        self.generations = 0
        self.pop_fit = []
        self.pop_sm = []

        import cma
        self.es = cma.CMAEvolutionStrategy(
            self.num_params * [0], self.sigma_init, {'popsize': self.pop_size})

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma*sigma))

    def ask(self):
        '''Returns a list of solutions'''
        self.solutions = np.array(self.es.ask())
        return torch.tensor(self.solutions)

    def tell(self, fitness_list):
        fitness_table = torch.tensor(np.array(fitness_list))
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(
                self.weight_decay, self.solutions).reshape(-1, 1)

            fitness_table += l2_decay
        # convert minimizer to maximizer:
        self.es.tell(self.solutions,
                     (-fitness_table).tolist())

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):
        '''return the best params so far along with historically best reward, current reward, sigma'''
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])

    def g_search(self, fitness_fn, env):
        # local trackers:
        best_train_fit = 1
        worst_train_fit = 1.0
        pop_avg = 1.0
        sm = 1.0
        caps_l = 1.0
        sm_sd = -1.0
        test_score = 1.0
        ep_len = 1.0
        ep_len_sd = -1.0

        self.generations += 1
        print("Generations: ", self.generations)
        solutions = self.ask()
        fx, sm, sm_sd, ep_len, ep_len_sd, caps_l, episode = fitness_fn(
            solutions, self.params, env)
        best_train_fit = fx.max().detach().numpy()
        worst_train_fit = fx.min().detach().numpy()
        pop_avg = fx.mean().detach().numpy()
        self.tell(fx)
        self.pop_fit.append(best_train_fit)
        self.pop_sm.append(sm)

        b_fx, _, _, _, _, _, last_episode = fitness_fn(
            torch.tensor(self.best_param()), self.params, env)
        test_score = b_fx[0].detach().numpy()
        if self.params.should_log:
            self.champion_history = last_episode.get_history()

        return {
            'best_train_fitness': best_train_fit,
            'test_score': test_score,
            'pop_avg': pop_avg,
            'pop_min': worst_train_fit,
            'avg_smoothness': sm,
            'caps_loss': caps_l,
            'smoothness_sd': sm_sd,
            'avg_ep_length': ep_len,
            'ep_length_sd': ep_len_sd
        }

    def search(self, fitness_fn, env):

        for _ in range(self.max_generations):
            self.generations += 1
            print("Generation: ", self.generations)
            solutions = self.ask()
            fx, sm_avg, sm_sd, _, _, caps_l, last_episode = fitness_fn(
                solutions, self.params, env)
            max_fit = fx.max().detach().numpy()
            print("maximum fitness:", max_fit)
            print("pop avg smoothness:", sm_avg)
            print("pop smoothness stdev:", sm_sd)
            self.pop_fit.append(max_fit)
            self.pop_sm.append(sm_avg)
            self.tell(fx)

        # champion test adn history:
        b_fx, _, _, _, _, _, last_episode = fitness_fn(
            torch.tensor(self.best_param()), self.params, env)
        if self.params.should_log:
            self.champion_history = last_episode.get_history()
        return self.best_param(), self.result()[1]

    def save_agent(self, parameters: ESParameters):
        """Save the trained agent (s)

        Args:
            parameters (object): the trained hyper-parameters:
        """
        b_agent = Actor(args=parameters, init=True)
        b_agent.inject_parameters(torch.tensor(self.best_param()))

        torch.save(b_agent.state_dict(), os.path.join(
            parameters.save_foldername, 'elite_agent.pkl'))

        if self.params.should_log:
            filename = 'statehistory_generation' + \
                str(self.generations) + '.txt'
            np.savetxt(os.path.join(parameters.save_foldername, filename),
                       self.champion_history, header=str(self.generations))

            print('>> Saved state history in ' + str(filename) + '\n')
