from itertools import chain
import gymnasium
import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from parameters_es import ESParameters
from evaluation_es_utils import Stats
from environments.config import select_env
from environments.aircraftenv import printPurple, printYellow
from core_algorithms.utils import Episode, calc_nMAE, calc_smoothness
from core_algorithms.random_process import OrnsteinUhlenbeckProcess
from core_algorithms.models import MLP
from core_algorithms.model_utils import GaussianNoise, soft_update
from core_algorithms.cem_rl import CEM
from copy import copy, deepcopy
from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import time

import ray
# ray.shutdown()
# ray.init()


class PPO(object):
    """Proximal Policy Optimization Algorithm

    Args:
        object (_type_): _description_
    """

    def __init__(self, args: ESParameters, a_noise: GaussianNoise or OrnsteinUhlenbeckProcess, env, env_name: str, use_cem: bool = False):
        print("Initializing PPO...")
        self.env = env
        self.env_name = env_name
        self.args = args
        self.noise = a_noise
        self.actor = MLP(args, is_actor=True)
        self.critic = MLP(args, is_actor=False)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=args.actor_lr)
        self.num_params = self.actor.count_parameters()
        print(">>> Number of Params: {}".format(self.num_params))
        self.use_cem = use_cem
        if use_cem:
            self.ESModel = CEM(
                num_params=self.num_params,
                params=args,
                sigma_init=args.sigma_init,
                mu_init=deepcopy(
                    self.actor.extract_parameters().cpu().numpy()),
                pop_size=args.pop_size,
                antithetic=not args.pop_size % 2,
                parents=args.parents,
                elitism=args.elitism,
                adaptation=args.cem_with_adapt,
            )
            self.es_dummy_actor = MLP(args, is_actor=True)

        self.action_std = self.args.init_action_std
        self.cov_var = torch.full(
            size=(self.args.action_dim,), fill_value=self.action_std**2).to(self.args.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.args.device)

        self.logging_stats = {
            'actor_loss': 0,
            'critic_loss': 0,
            'entropy_loss': 0,
            'approx_kl': 0,
            'rl_avg_score': 0,
            # 'rl_sm': 0,
            # 'rl_ep_length': 0,
            # 'avg_smoothness': 0,
            # 'smoothness_sd': 0,
            # 'avg_ep_length': 0,
            # 'ep_length_sd': 0,
            # 'total_steps': 0,
            'test_best_mu_so_far_score': 0,
            'test_best_elite_so_far_score': 0,
            'test_mu_score': 0,
            'pop_min': 0,
            'pop_avg': 0,
            'best_train_fitness': 0,
            'sigma_value': 0,
            'time': 0,
        }

    def decay_action_std(self):
        self.action_std *= self.args.action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        self.action_std = max(self.action_std, self.args.action_std_min)
        self.cov_var = torch.full(
            size=(self.args.action_dim,), fill_value=self.action_std**2).to(self.args.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.args.device)

        print('>> Setting output action std to {}'.format(self.action_std))

    def take_action(self, obs):
        """Select action based on observation

        Args:
            obs (tensor): observation from the environment

        Returns:
            np.ndarray: action to take
        """
        # if es_eval:
        #     print(">> Evaluate ES candidate ")
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.args.device)
            mean = self.actor(obs)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            action = np.clip(action, self.args.action_low,
                             self.args.action_high)
            action_log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy().flatten(), action_log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """_summary_

        Args:
            batch_obs (_type_): _description_
            batch_acts (_type_): _description_
        """
        # using most recent critic to obtain state values:
        V = self.critic(batch_obs).squeeze()
        # using most recent actor to obtain action log probs:
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action_log_probs = dist.log_prob(batch_acts)

        return V, action_log_probs, dist.entropy()

    def calculate_gae(self, rewards, values, dones):
        """_summary_

        Args:
            rewards (_type_): _description_
            values (_type_): _description_
            dones (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_adv = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advs = []
            last_adv = 0
            for t in reversed(range(len(ep_rews))):
                if t+1 < len(ep_rews):
                    delta = ep_rews[t] + self.args.gamma * \
                        ep_vals[t+1] * (1-(ep_dones[t])) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                adv = delta + self.args.gamma * \
                    self.args.lam * last_adv * (1-(ep_dones[t]))
                last_adv = adv
                advs.insert(0, adv)
            batch_adv.extend(advs)
        batch_adv = torch.FloatTensor(batch_adv).to(self.args.device)
        return batch_adv

    def rollout(self, es_sol):
        batch_obs = []
        batch_acts = []
        batch_dones = []
        batch_state_values = []
        batch_log_probs = []
        batch_rewards = []
        batch_episode_lengths = []

        t = 0
        # has_episode_end = 0
        while t < self.args.timesteps_per_batch:
            ep_rewards = []
            ep_state_values = []
            ep_dones = []

            obs, _ = self.env.reset()
            done = False
            truncated = False

            for ep_t in range(self.args.max_timesteps_per_episode):

                batch_obs.append(obs)
                ep_dones.append(float(done))

                action, action_log_prob = self.take_action(
                    obs)
                t += 1
                val = self.critic(obs)
                obs, reward, done, truncated, _ = self.env.step(action)

                batch_acts.append(action)
                ep_rewards.append(reward)
                ep_state_values.append(val)
                batch_log_probs.append(action_log_prob)

                if done or truncated:
                    break

            batch_rewards.append(ep_rewards)
            batch_episode_lengths.append(ep_t+1)
            batch_state_values.append(ep_state_values)
            batch_dones.append(ep_dones)

            if es_sol is not None:
                ad_obs, ad_acts, ad_log, ad_val, ad_rews, ad_done, ad_epl = zip(*ray.get([single_agent_rollout_remote.remote(
                    theta, self.env_name, self.critic.extract_parameters().cpu().numpy(), self.args, self.cov_mat) for theta in es_sol]))

                ad_epl = list(chain(*ad_epl))
                ad_obs = list(chain(*ad_obs))
                ad_acts = list(chain(*ad_acts))
                ad_log = list(chain(*ad_log))
                ad_val = list(ad_val)
                print(len(ad_val))
                t += sum(ad_epl)
                batch_obs.extend(ad_obs)
                batch_acts.extend(ad_acts)
                batch_log_probs.extend(ad_log)
                for i in range(self.args.pop_size):
                    batch_state_values.extend(ad_val[i])
                    batch_rewards.extend(ad_rews[i])
                    batch_dones.extend(ad_done[i])
                batch_episode_lengths.extend(ad_epl)

        batch_obs = torch.FloatTensor(
            np.asarray(batch_obs)).to(self.args.device)
        batch_acts = torch.FloatTensor(
            np.asarray(batch_acts)).to(self.args.device)
        batch_log_probs = torch.FloatTensor(
            batch_log_probs).to(self.args.device)

        return batch_obs, batch_acts, batch_log_probs, batch_state_values, batch_rewards, batch_dones, batch_episode_lengths

    def learn(self, total_timesteps):
        """_summary_

        Args:
            total_timesteps (_type_): _description_
        """
        t_so_far = 0
        iter_so_far = 0
        # threshold = -np.inf
        start_time = time.time()
        while t_so_far < total_timesteps:
            printYellow(
                f'>>> Timesteps {t_so_far} - Iteration {iter_so_far} <<<')
            # evaluate the ppo actor:
            actor_fit = self.eval_rl_actor()
            printYellow(">> PPO Actor Fit: {}".format(actor_fit))
            es_sol = None
            if self.args.use_cem:
                es_sol = self.ESModel.ask(self.args.pop_size)
                self.update_es(es_sol, noisy=False)
                self.es_dummy_actor.inject_parameters(
                    self.ESModel.best_mu_so_far)

                if actor_fit > self.ESModel.rl_agent_score:
                    self.ESModel.rl_agent_score = actor_fit
                    self.ESModel.rl_agent = self.actor.extract_parameters().cpu().numpy()
                self.logging_stats['test_best_mu_so_far_score'] = self.ESModel.best_mu_so_far_score
                self.logging_stats['test_best_elite_so_far_score'] = self.ESModel.best_elite_so_far_score

                if iter_so_far % 2 == 0:
                    if actor_fit < self.ESModel.best_mu_so_far_score:
                        printPurple(
                            '>> Replace PPO actor with ES Best MU actor')
                        self.actor.inject_parameters(
                            self.ESModel.best_mu_so_far)
                    else:
                        printPurple('>> Put PPO actor as ES Mu')
                        self.ESModel.mu = deepcopy(
                            self.actor.extract_parameters().cpu().numpy())
                        # self.es_dummy_actor.inject_parameters(
                        #     self.ESModel.best_mu_so_far)
                        # soft_update(
                        #     self.actor, self.es_dummy_actor, self.args.tau)
            self.logging_stats['rl_avg_score'] = actor_fit
            # self.logging_stats['rl_sm'] = actor_sm
            # self.logging_stats['rl_ep_length'] = actor_ep

            # TODO: replace ppo actor with ES actor:

            # collect trajectories:
            batch_obs, batch_acts, batch_log_probs, batch_state_values, batch_rewards, batch_dones, batch_episode_lengths = self.rollout(
                es_sol)

            print(">> Batch Obs: ", batch_obs.shape)

            self.decay_action_std()
            # update number of steps so far and iteration:
            self.logging_stats['total_steps'] = t_so_far
            t_so_far += sum(batch_episode_lengths)
            self.logging_stats['total_iterations'] = iter_so_far
            iter_so_far += 1

            # calculate general advantages:
            A_k = self.calculate_gae(
                batch_rewards, batch_state_values, batch_dones)
            # calculate Value V_k:
            V_k = self.critic(batch_obs).squeeze()
            # calculate return to go:
            batch_returns_to_go = A_k + V_k.detach()

            # normalizing the advantages:
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # train actor and critic via minibatches:
            N = batch_obs.shape[0]
            inds = np.arange(N)
            minibatch_size = N // self.args.num_minibatches

            logged_loss = []

            for _ in tqdm(range(self.args.n_updates_per_iteration), desc='PPO Update'):
                # learning rate update:
                frac = 1.0 - float(t_so_far-1) / total_timesteps
                curr_lr = self.args.actor_lr * frac
                curr_lr = max(curr_lr, 0.0)
                for g in self.actor_optimizer.param_groups:
                    g['lr'] = curr_lr
                for g in self.critic_optimizer.param_groups:
                    g['lr'] = curr_lr

                # shuffle the indices:
                np.random.shuffle(inds)
                for start in range(0, N, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    minibatch_obs, minibatch_acts, minibatch_logprobs, minibatch_rtgs, minibatch_advantages = batch_obs[
                        idx], batch_acts[idx], batch_log_probs[idx], batch_returns_to_go[idx], A_k[idx]

                    # current V and log prob and entropy:
                    V, new_log_prob, entropy = self.evaluate(
                        minibatch_obs, minibatch_acts)

                    # ratios of probs:
                    log_ratios = new_log_prob - minibatch_logprobs
                    ratios = torch.exp(log_ratios)
                    approx_kl = ((ratios-1)-log_ratios).mean()

                    # surrogate loss:
                    surr1 = ratios * minibatch_advantages
                    surr2 = torch.clamp(
                        ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * minibatch_advantages

                    # actor and critic loss:
                    entropy_loss = entropy.mean()
                    actor_loss = - \
                        torch.min(surr1, surr2).mean() - \
                        self.args.entropy_coef * entropy_loss
                    critic_loss = nn.MSELoss()(V, minibatch_rtgs)

                    # gradient updates:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.args.max_grad_norm)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.args.max_grad_norm)
                    self.critic_optimizer.step()

                    logged_loss.append([actor_loss.detach(), critic_loss.detach(
                    ), entropy_loss.detach(), approx_kl.detach()])

                if approx_kl > self.args.target_kl:
                    print('>> Early stopping at iteration {} due to reaching max kl.'.format(
                        iter_so_far))
                    break
            avg_losses = torch.tensor(logged_loss).mean(axis=0)
            self.logging_stats['actor_loss'] = avg_losses[0]
            self.logging_stats['critic_loss'] = avg_losses[1]
            self.logging_stats['entropy_loss'] = avg_losses[2]
            self.logging_stats['approx_kl'] = avg_losses[3]

            self.logging_stats['time'] = time.time() - start_time

            if self.args.should_log:
                wandb.log(self.logging_stats)

            print(self.logging_stats)
            # print('>> Iteration {} \n PPO Actor loss: {:.3f} \n PPO Critic loss: {:.3f} \n PPO Entropy loss: {:.3f} \n PPO Approx KL: {:.5f} \n ES Best Mu Fit: {:.2f} \n Best Elite Fit: {:.2f} '.format(
            #     iter_so_far, avg_losses[0], avg_losses[1], avg_losses[2], avg_losses[3], self.ESModel.best_mu_so_far_score, self.ESModel.best_elite_so_far_score))

    def eval_trained_agent(self, actor, deterministic: bool = True, **kwargs):
        """_summary_

        Args:
            deterministic (bool, optional): _description_. Defaults to True.
        """

        x_ctrl_lst = []
        errors = []
        ref_lst = []
        t_lst = []

        state_lst, rewards, action_lst = [], [], []
        obs, _ = self.env.reset(**kwargs)
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            action = actor.act(
                obs) if deterministic else self.take_action(obs)[0]
            action = np.clip(action, self.args.action_low,
                             self.args.action_high)
            ref_value = np.deg2rad(
                np.array([ref(self.env.t) for ref in self.env.ref]).flatten())
            obs, reward, done, truncated, info = self.env.step(action)
            action_lst.append(self.env.last_u)
            state_lst.append(self.env.x)
            x_ctrl_lst.append(self.env.get_controlled_state())

            steps += 1

            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(self.env.t)

            if done:
                self.env.reset()

        self.env.close()

        errors = np.asarray(errors)
        actions = np.array(action_lst)
        smoothness = calc_smoothness(actions, plot_spectra=False)
        fitness = np.sum(rewards) + smoothness
        # if fitness >= threshold:
        #     threshold = fitness

        episode = Episode(
            fitness=fitness,
            smoothness=smoothness,
            n_steps=steps,
            length=info['t'],
            state_history=state_lst,
            ref_signals=info['ref'],
            actions=actions,
            reward_lst=rewards,
            threshold=-np.inf,
        )

        # format data:
        rewards = np.asarray(rewards).reshape((-1, 1))
        ref_values = np.array(ref_lst)
        t_lst = np.asarray(t_lst).reshape((-1, 1))
        data = np.concatenate(
            (ref_values, actions, state_lst, rewards, t_lst), axis=1)

        # calculate nMAE:
        nmae = calc_nMAE(errors)
        return episode, data, nmae, smoothness

    def eval_rl_actor(self):
        s, _ = self.env.reset()
        done = False
        truncated = False
        reward = 0.0
        actions = []
        for step in range(self.args.max_timesteps_per_episode):
            a = self.actor.act(s)
            a = np.clip(a, self.args.action_low, self.args.action_high)
            s, r, done, truncated, _ = self.env.step(a)
            reward += r*(self.args.gamma**step)
            if self.env_name.startswith('PH'):
                actions.append(self.env.last_u)
            else:
                actions.append(a)

            if done or truncated:
                break

        self.env.close()
        if self.env_name.startswith('PH'):
            return reward + calc_smoothness(np.asarray(actions))
        return reward

    def avg_fitness_over_n_evals(self, actor, deterministic: bool = True, n_evals=3):
        """_summary_

        Args:
            env (_type_): _description_
            deterministic (bool, optional): _description_. Defaults to True.
            threshold (_type_, optional): _description_. Defaults to -np.inf.
        Returns:
            fitness, smoothness, sm_sd, episode_length, ep_sd, steps, episode
        """
        ft_scores, sm_scores, ep_lengths = [], [], []
        total_steps = 0
        for _ in range(n_evals):
            episode, _, _, _ = self.eval_trained_agent(
                actor, deterministic)
            ft_scores.append(episode.fitness)
            sm_scores.append(episode.smoothness)
            ep_lengths.append(episode.length)

            total_steps += episode.n_steps

        actor_fit = np.array(ft_scores).mean()
        sm_scores = np.array(sm_scores)
        sm_avg = np.median(sm_scores)
        sm_sd = np.std(sm_scores)
        ep_length_avg = np.array(ep_lengths).mean()
        ep_length_sd = np.array(ep_lengths).std()

        return actor_fit, sm_avg, sm_sd, ep_length_avg, ep_length_sd, total_steps, episode

    def eval_on_trajectory(self, actor, user_ref_lst: list, num_trials: int = 1, deterministic: bool = True, **kwargs):
        """_summary_

        Args:
            user_ref_lst (list): _description_
            num_trials (int, optional): _description_. Defaults to 1.
            deterministic (bool, optional): _description_. Defaults to True.

        """
        agent_nmae_lst, agent_sm_lst = [], []
        for i in tqdm(range(num_trials+1), total=num_trials):
            ref_t = user_ref_lst[i]
            user_refs = {
                'theta_ref': ref_t[0],
                'phi_ref': ref_t[1],
                # 'psi_ref': ref_t[2],
            }

            _, data, nmae, sm = self.eval_trained_agent(
                actor, deterministic, user_refs=user_refs, **kwargs)
            agent_nmae_lst.append(nmae)
            agent_sm_lst.append(sm)

        nmae = np.mean(agent_nmae_lst)
        nmae_std = np.std(agent_nmae_lst)
        smoothness = np.median(agent_sm_lst)
        sm_std = np.std(agent_sm_lst)

        stats = Stats(nmae, nmae_std, smoothness, sm_std)
        return data, stats

    def update_es(self, es_sol, noisy: bool = False):

        # with Pool() as p:
        #     fit_lst = p.map(partial(eval_es_sol, args=self.args), es_sol)

        # TODO with ray:
        fit_lst = ray.get([eval_es_sol_remote.remote(
            theta, self.args, self.env_name, noisy) for theta in es_sol])

        fit_lst = np.array(fit_lst)
        self.calculate_logging_stats(
            fit_lst)
        ################
        self.ESModel.tell(es_sol, fit_lst)
        # self.es_dummy_actor.inject_parameters(self.ESModel.mu)
        test_mu = eval_es_sol(self.ESModel.mu, args=self.args,
                              env_name=self.env_name, noisy=False)
        self.logging_stats['test_mu_score'] = test_mu
        if test_mu > self.ESModel.best_mu_so_far_score:
            self.ESModel.best_mu_so_far = deepcopy(self.ESModel.mu)
            self.ESModel.best_mu_so_far_score = test_mu
            printPurple('>> New best mu so far: {}'.format(test_mu))

        return fit_lst

    def calculate_logging_stats(self, fit_lst):
        # prepare for statistics:
        self.logging_stats['best_train_fitness'] = np.array(fit_lst).max()
        self.logging_stats['pop_min'] = np.array(fit_lst).min()
        self.logging_stats['pop_avg'] = np.array(fit_lst).mean()
        # self.logging_stats['avg_smoothness'] = np.array(smAvg_lst).mean()
        # self.logging_stats['smoothness_sd'] = np.array(smSd_lst).mean()
        # self.logging_stats['avg_ep_length'] = np.array(epAvg_lst).mean()
        # self.logging_stats['ep_length_sd'] = np.array(epSd_lst).mean()
        self.logging_stats['sigma_value'] = self.ESModel.sigma

    def parallel_fitness_computation(self, controller, deterministic: bool = True):

        fit, sm, sm_sd, ep, ep_sd, _, _ = self.avg_fitness_over_n_evals(
            controller, deterministic=deterministic, n_evals=1)
        return fit, sm, sm_sd, ep, ep_sd

    def load_agent(self, model_path, args, agent_type_arg):
        model_path = model_path / Path('files')

        rl_actor = 'rl_agent.pkl'
        rl_actor_ckpt = torch.load(os.path.join(model_path, rl_actor))
        self.actor.load_state_dict(rl_actor_ckpt)

        rl_critic = 'rl_critic.pkl'
        rl_critic_ckpt = torch.load(os.path.join(model_path, rl_critic))
        self.critic.load_state_dict(rl_critic_ckpt)

        if args.continue_training:
            # load COV, mu, best_mu and best_elite for ES:
            self.ESModel.cov = np.loadtxt(
                os.path.join(model_path, 'best_cov.csv'))
            # print('Loaded COV for ES', self.ESModel.cov)
            self.es_dummy_actor.load_state_dict(
                torch.load(os.path.join(model_path, 'best_mu_agent.pkl')))
            self.ESModel.mu = self.es_dummy_actor.extract_parameters(
            ).cpu().numpy()  # best mu loaded as Mu
            self.ESModel.mu_score = eval_es_sol(
                self.ESModel.mu, args=self.args, noisy=False)
            self.ESModel.best_mu_so_far_score = copy(self.ESModel.mu_score)
            self.es_dummy_actor.load_state_dict(torch.load(
                os.path.join(model_path, 'best_mu_agent.pkl')))
            self.ESModel.best_mu_so_far = self.es_dummy_actor.extract_parameters().cpu().numpy()
            self.es_dummy_actor.load_state_dict(torch.load(
                os.path.join(model_path, 'best_elite_agent.pkl')))
            self.ESModel.best_elite_so_far = self.es_dummy_actor.extract_parameters().cpu().numpy()
            printPurple('>> Current Mu score: {}'.format(
                self.ESModel.mu_score))

        if agent_type_arg.use_mu:
            print('Using Mu agent')
            agent_name = 'mu_agent.pkl'
        elif agent_type_arg.use_best_mu:
            print('Using Best Mu agent seen')
            agent_name = 'best_mu_agent.pkl'
        elif agent_type_arg.use_best_elite:
            print('Using Best Elite agent seen')
            agent_name = 'best_elite_agent.pkl'
        else:
            print('Using Elite of last gen')
            agent_name = 'elite_agent.pkl'

        self.es_dummy_actor.load_state_dict(
            torch.load(os.path.join(model_path, agent_name)))

        print("Agents loaded from: " + str(model_path))


def eval_es_sol(theta, args, env_name, noisy: bool = False):
    policy = MLP(args=args, is_actor=True)
    policy.inject_parameters(theta)
    # max_steps = 2000
    if env_name.startswith('PH'):
        env = select_env(
            environment_name=args.env_name,
            conform_with_sb=True,
        )
    else:
        env = gymnasium.make(env_name, g=9.81)
    noise = GaussianNoise(
        action_dimension=args.action_dim,
        std=args.gauss_sigma
    )
    s, _ = env.reset()
    done, truncated = False, False
    total_reward = 0.0
    actions = []
    for step in range(args.max_timesteps_per_episode):
        a = policy.act(s)

        if noisy:
            n_val = np.clip(noise.noise(), -args.noise_clip, args.noise_clip)
            a += n_val
        a = np.clip(a, args.action_low, args.action_high)
        s, r, done, truncated, _ = env.step(a)
        if env_name.startswith('PH'):
            actions.append(env.last_u)
        else:
            actions.append(a)
        total_reward += r*(args.gamma**step)

        if done or truncated:
            break
    env.close()
    if env_name.startswith('PH'):
        return total_reward + calc_smoothness(np.asarray(actions))

    return total_reward


@ray.remote
def eval_es_sol_remote(theta, args, env_name, noisy):
    return eval_es_sol(theta, args, env_name, noisy)


def single_agent_rollout(theta, env_name, critic_params, args, cov_mat):
    policy = MLP(args=args, is_actor=True)
    policy.inject_parameters(theta)
    critic = MLP(args, is_actor=False)
    critic.inject_parameters(critic_params)
    # max_steps = 2000
    if env_name.startswith('PH'):
        env = select_env(
            environment_name=args.env_name,
            conform_with_sb=True,
        )
    else:
        env = gymnasium.make(env_name, g=9.81)
    s, _ = env.reset()
    done, truncated = False, False
    steps = 0
    ep_rewards = []
    ep_state_values = []
    ep_dones = []
    agent_batch_obs = []
    agent_batch_acts = []
    agent_batch_log_probs = []
    agent_batch_rewards = []
    agent_batch_episode_lengths = []
    agent_batch_state_values = []
    agent_batch_dones = []
    while steps < args.max_timesteps_per_episode:
        agent_batch_obs.append(s)
        ep_dones.append(float(done))
        with torch.no_grad():
            if isinstance(s, np.ndarray):
                s = torch.FloatTensor(s).to(args.device)
            mean = policy(s)
            dist = MultivariateNormal(mean, cov_mat)
            a = dist.sample()
            a_log_p = dist.log_prob(a).detach()
            a = a.detach().cpu().numpy().flatten()

        val = critic(s)
        s, r, done, truncated, _ = env.step(a)
        ep_rewards.append(r)
        ep_state_values.append(val)
        agent_batch_acts.append(a)
        agent_batch_log_probs.append(a_log_p)

        steps += 1
        if done or truncated:
            break

    agent_batch_rewards.append(ep_rewards)
    agent_batch_episode_lengths.append(steps)
    agent_batch_state_values.append(ep_state_values)
    agent_batch_dones.append(ep_dones)

    return agent_batch_obs, agent_batch_acts, agent_batch_log_probs, agent_batch_state_values, agent_batch_rewards, agent_batch_dones, agent_batch_episode_lengths


@ray.remote
def single_agent_rollout_remote(theta, env_name, critic_params, args, cov_mat):
    return single_agent_rollout(theta, env_name, critic_params, args, cov_mat)
