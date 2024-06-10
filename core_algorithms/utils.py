import torch
from tqdm import tqdm
from core_algorithms.multi_agent import MultiAgentActor
from parameters_es import ESParameters
from core_algorithms.replay_memory import ReplayMemory
from core_algorithms.some_actor_model import CONV_ACTOR, Actor, LSTM_ACTOR
from core_algorithms.sac import SACGaussianActor, SACDeterministicActor, SAC
from core_algorithms.td3 import TD3ES
import os
from dataclasses import dataclass
from pprint import pprint
import numpy as np
from typing import List
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


@dataclass
class Episode:
    fitness: np.float64
    smoothness: np.float64
    # caps_loss: np.float64
    n_steps: int
    length: np.float64
    state_history: List
    ref_signals: List
    actions: List
    reward_lst: List
    threshold: np.float64

    def get_history(self):
        """
        Returns the time traced state history of the episode.
        """
        time_trace_hist = np.linspace(0, self.length, len(self.state_history))
        ref_values = np.array([[np.deg2rad(ref(t_i)) for t_i in time_trace_hist]
                              for ref in self.ref_signals]).transpose()
        # print(ref_values)
        reward_lst = np.asarray(self.reward_lst).reshape(
            (len(self.state_history), 1))

        return np.concatenate((ref_values, self.actions, self.state_history, reward_lst), axis=1)


def calc_nMAE(error: np.array):
    """Calculates the normalized Mean absolute error using the error time history
    Args:
        error (np.array): error time history
    Returns:
        float: normalized Mean absolute error in percentage
    """
    n = error.shape[1]
    # print(n)
    mae = np.mean(np.abs(error), axis=0)
    theta_range = np.deg2rad(20)
    phi_range = np.deg2rad(20)
    beta_range = max(
        np.abs(np.average(error[:, -1])), 3.14159/100) if n == 3 else None
    if n == 3:
        signal_range = np.array([theta_range, phi_range, beta_range])
    else:
        signal_range = np.array([theta_range, phi_range])

    nmae = mae/signal_range
    return np.mean(nmae) * 100  # normalized (in percentage)


def calc_nMAE_from_ref(ref: np.ndarray, x_crtl: np.ndarray):
    """Calculates the normalized Mean absolute error using the error time history
    Args:
        ref (np.array): ref signal
        x_ctrl (np.array): controlled state signal
    Returns:
        float: normalized Mean absolute error in percentage
    """
    error = ref - x_crtl
    mae = np.mean(np.abs(error), axis=0)
    signal_range = np.deg2rad(np.array([20, 20, 1]))
    nmae = mae/signal_range
    print(np.average(error, axis=0))
    print(f"Theta, phi, beta: {nmae*100} -> total: {100*np.mean(nmae)}\n")
    return np.mean(nmae) * 100  # normalized (in percentage)


def calc_smoothness(y: np.ndarray, dt: float = 0.01, **kwargs):
    """Calculates the smoothness of a signal
    Args:
        y (np.ndarray): signal
        dt (float, optional): time step. Defaults to 0.01.
    Returns:
        float: smoothness of the signal
    """

    def _roughness(seq):
        N = seq.shape[0]
        A = seq.shape[1]
        T = N * dt
        freq = np.linspace(dt, 1/(2*dt), N//2 - 1)
        Syy = np.zeros((N // 2 - 1, A))
        for i in range(A):
            Y = fft(seq[:, i], N)
            Syy_disc = Y[1:N//2] * np.conjugate(Y[1:N//2])  # discrete
            Syy[:, i] = np.abs(Syy_disc) * dt   # continuous

        # Smoothness of each signal
        signal_roughness = np.einsum('ij,i -> j', Syy, freq) * 2/N
        _S = np.sum(signal_roughness, axis=-1)
        roughness = np.sqrt(_S) * 100 * (80/T)
        if kwargs.get('plot_spectra'):
            plt.figure(num='spectra')
            plt.title(
                f'Spectra actuating Signals\n Smoothness: {-roughness:0.0f}')
            plt.loglog(freq, Syy[:, 0] * 2/N,
                       linestyle='-', label=r'$\delta_e$')
            plt.loglog(freq, Syy[:, 1] * 2/N,
                       linestyle='--', label=r'$\delta_a$')
            plt.loglog(freq, Syy[:, 2] * 2/N,
                       linestyle=':', label=r'$\delta_r$')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude')
            plt.legend(loc='best')
        return roughness

    # assert the dim of signal y: if dim<2 then reshape:

    if not len(y.shape) > 1:
        y = y[:, np.newaxis]

    in_shape = y.shape
    assert len(in_shape) > 1, "Signal should be at least of dim 2"
    if len(y.shape) > 2:
        y = y.transpose(1, 0, 2)

        n_env = y.shape[0]
        envs_s = []
        for e in range(n_env):

            s_e = _roughness(y[e])
            envs_s.append(s_e)
        # roughness = -np.mean(envs_s)
        roughness = np.asarray(envs_s)
        # print(roughness)
    else:
        roughness = _roughness(y)

    return -roughness


def load_config(model_path: str, verbose: bool = False):
    """ Load controller configuration from file.

    Args:
        model_path (str): Absolute path to logging folder.

    Returns:
        dict: Configuration dictionary.
    """
    model_path = model_path / Path('files/')
    config_path = os.path.join(model_path, 'config.yaml')
    conf_raw = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))

    conf_dict = {}
    for k, _v in conf_raw.items():
        if isinstance(_v, dict):
            value = _v['value']
        else:
            value = _v
        conf_dict[k] = value

    if verbose:
        pprint(conf_dict)
    return conf_dict


def train_rl(agent: TD3ES or SAC, params: ESParameters, g_buffer: ReplayMemory = None, b_buffer: ReplayMemory = None, n_buffer: ReplayMemory = None, rl_iter=0, rl_transitions=0):
    """Compute the score of an actor on a given environment and fills the memory when needed

    Args:
        actor (Actor or RNN_Actor): _description_
        env (PHlab, LunarLander): _description_
        params (ESParameters): _description_
        buffer (ReplayMemory, optional): _description_. Defaults to None.
        random (bool): _description_. Defaults to False.
        noise (sd): _description_. Defaults to None
    """
    # **** batch training:
    policy_grad_loss, TD_loss = [], []  # TODO: the policy gradient loss and TD error
    use_noise = False
    if g_buffer.__len__() >= params.start_steps and b_buffer.__len__() >= params.start_steps:
        if n_buffer.__len__() >= params.start_steps:
            use_noise = True
        agent.actor.train()
        for _ in tqdm(range(rl_transitions), desc='RL training', colour='blue'):
            rl_iter += 1
            g_batch = g_buffer.sample(params.g_batch_size)
            b_batch = b_buffer.sample(params.b_batch_size)
            if use_noise:
                n_batch = n_buffer.sample(params.n_batch_size)
            else:
                # sample again from the good or elite buffer:
                n_batch = g_buffer.sample(params.n_batch_size)

            state = torch.cat((g_batch[0], b_batch[0], n_batch[0]))
            action = torch.cat((g_batch[1], b_batch[1], n_batch[1]))
            next_state = torch.cat(
                (g_batch[2], b_batch[2], n_batch[2]))
            reward = torch.cat((g_batch[3], b_batch[3], n_batch[3]))
            done = torch.cat((g_batch[4], b_batch[4], n_batch[4]))
            comb_batch = (state, action, next_state, reward, done)

            pgl, td_l = agent.update_parameters(comb_batch, rl_iter)

            if pgl is not None:
                policy_grad_loss.append(-pgl)
            if td_l is not None:
                TD_loss.append(td_l)
    # update the observation:

    # reset when done:

    return rl_iter, np.mean(TD_loss)


def evaluate(actor: Actor or LSTM_ACTOR or CONV_ACTOR or SACGaussianActor or SACDeterministicActor or MultiAgentActor, env, params: ESParameters, g_buffer: ReplayMemory = None, b_buffer: ReplayMemory = None, n_buffer: ReplayMemory = None, random=False, noise=None, threshold=-np.inf):
    """Compute the score of an actor on a given environment and fills the memory when needed

    Args:
        actor (Actor or RNN_Actor): _description_
        env (PHlab, LunarLander): _description_
        params (ESParameters): _description_
        buffer (ReplayMemory, optional): _description_. Defaults to None.
        random (bool): _description_. Defaults to False.
        noise (sd): _description_. Defaults to None
    """
    # print(threshold)
    if not random:
        def policy(obs):
            action = actor.select_action(obs)
            if noise is not None:
                action += noise.sample()
            return np.clip(action, -1.0, 1.0)
    else:
        def policy(obs):
            return env.action_space.sample()

    state_lst, rewards, action_lst = [], [], []
    done = False
    # reset the environment:
    obs, _ = env.reset()
    actor.eval()
    steps = 0
    swap_buffer = ReplayMemory(params.mem_size)
    while not done:
        # get the action from the actor:
        action = policy(obs)
        # take a step in the environment:
        if 'lunar' in params.env_name:
            next_obs, reward, done, truncated, _ = env.step(
                action.flatten())
            action_lst.append(action.flatten())
            state_lst.append(obs)
        else:
            next_obs, reward, done, info = env.step(action.flatten())
            action_lst.append(env.last_u)
            state_lst.append(env.x)
        # update the score:
        rewards.append(reward)
        steps += 1

        # TODO: Elite buffer or multi-buffer
        if g_buffer is not None or b_buffer is not None or n_buffer is not None:
            if params.use_state_history:
                transition = (np.expand_dims(obs, axis=0),
                              action, np.expand_dims(next_obs, axis=0), reward, float(done))
            else:
                transition = (obs, action, next_obs, reward, float(done))
            swap_buffer.push(*transition)
        # TODO: with or without noise
        if n_buffer is not None and noise is not None:
            n_buffer.push(*transition)

        # update the observation:
        obs = next_obs

        # reset when done:
        if done:
            env.reset()
    env.close()
    actions = np.array(action_lst)
    smoothness = calc_smoothness(actions, plot_spectra=False)
    fitness = np.sum(rewards) + smoothness
    num_traj = len(rewards)
    if fitness >= 0.9*threshold:
        if g_buffer is not None:
            g_buffer.add_latest_from(swap_buffer, num_traj)
        if fitness > threshold:
            threshold = fitness
    else:
        if b_buffer is not None:
            b_buffer.add_latest_from(swap_buffer, num_traj)

    # print(g_buffer.__len__(), b_buffer.__len__(), n_buffer.__len__())

    swap_buffer.reset()
    if 'lunar' not in params.env_name:
        episode = Episode(
            fitness=fitness,
            smoothness=smoothness,
            n_steps=steps,
            length=info['t'],
            state_history=state_lst,
            ref_signals=info['ref'],
            actions=actions,
            reward_lst=rewards,
            threshold=threshold
        )
    else:
        episode = Episode(
            fitness=fitness,
            smoothness=smoothness,
            n_steps=steps,
            actions=actions,
            reward_lst=rewards,
            state_history=state_lst,
            threshold=threshold
        )
    return episode


def fitness_function(actor: Actor or LSTM_ACTOR or CONV_ACTOR or SACGaussianActor or SACDeterministicActor or MultiAgentActor, env, params: ESParameters, g_buffer: ReplayMemory = None, b_buffer: ReplayMemory = None, n_buffer: ReplayMemory = None, random=False, noise=None, n_evals=3, threshold=-np.inf):
    """_summary_

    Args:
        actor (Actor or LSTM_ACTOR): _description_
        env (_type_): _description_
        params (ESParameters): _description_
        buffer (ReplayMemory, optional): _description_. Defaults to None.
        random (bool, optional): _description_. Defaults to False.
        noise (_type_, optional): _description_. Defaults to None.
        n_evals (int, optional): _description_. Defaults to 1.
    """
    ft_scores, sm_scores, ep_lengths, thresholds = [], [], [], []
    total_steps = 0
    for _ in range(n_evals):
        episode = evaluate(actor, env, params, g_buffer, b_buffer, n_buffer,
                           random, noise, threshold=threshold)
        ft_scores.append(episode.fitness)
        sm_scores.append(episode.smoothness)
        ep_lengths.append(episode.length)
        thresholds.append(episode.threshold)
        total_steps += episode.n_steps

    actor_fit = np.array(ft_scores).mean()
    sm_scores = np.array(sm_scores)
    sm_avg = np.median(sm_scores)
    sm_sd = np.std(sm_scores)
    ep_length_avg = np.array(ep_lengths).mean()
    ep_length_sd = np.array(ep_lengths).std()

    # print(thresholds)
    thresholds = np.array(thresholds)
    # print(np.min(thresholds), np.max(thresholds))
    n_threshold = np.max(thresholds)
    if actor_fit >= n_threshold:
        n_threshold = actor_fit

    return actor_fit, sm_avg, sm_sd, ep_length_avg, ep_length_sd, total_steps, episode, n_threshold


if __name__ == '__main__':
    N = 2000
    dt = 0.01
    x = np.linspace(0.0, N*dt, N, endpoint=False)
    y = np.zeros_like(x)

    clipped_noise = np.clip(0.3 * np.random.randn(y.shape[0]), - 0.5, 0.5)
    _action = np.clip(x + clipped_noise, -1.0, 1.0)
    action = np.array([_action, _action, _action]).reshape(-1, 3)

    smoothness = calc_smoothness(action, plot_spectra=True)

    print(smoothness * N*dt)

    plt.show()
