# import matplotlib.font_manager as font_manager
# import time
# import warnings
import toml
from copy import deepcopy, copy
# import os
from pathlib import Path
from evaluation_es_utils import find_logs_path, gen_refs, Stats, load_agent
from signals.sequences import SmoothedStepSequence
import random
import argparse
import numpy as np
from core_algorithms.utils import calc_nMAE, calc_smoothness, load_config
from parameters_es import ESParameters
# import matplotlib.pyplot as plt

from environments.config import select_env
from tqdm import tqdm

import torch
from plotters.plot_utils import plot, plot_comparative_analysis


parser = argparse.ArgumentParser()

############ Arguments #############
parser.add_argument('--env-name', help='Environment to be used: (PHLab)',
                    type=str, default='PHlab_attitude_nominal')
parser.add_argument('--use-second-env', action='store_true', default=False)
parser.add_argument('--env2-name', help='Environment 2 to be used: (PHLab)',
                    type=str, default='PHlab_attitude_be')
parser.add_argument('--switch-time', type=float, default=20.0)
parser.add_argument('--use-state-history', action='store_true')
parser.add_argument(
    '--use-mu', help='Use mu instead of best agent', action='store_true')
parser.add_argument('--use-best-mu', action='store_true')
parser.add_argument('--use-best-elite', action='store_true')
parser.add_argument('--disable-cuda', help='Disables CUDA',
                    action='store_true', default=False)
parser.add_argument('--num-trials', type=int, default=2)
parser.add_argument(
    '--agent-name', help='Path to the agent to be evaluated', type=str, default='CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed0')
parser.add_argument('--seed', help='Random seed to be used',
                    type=int, default=7)

parser.add_argument('--t-max', type=int, default=80)
parser.add_argument('--plot-spectra', default=False, action='store_true')
parser.add_argument('--save-plots', default=False, action='store_true')
parser.add_argument('--save-trajectory', default=False, action='store_true')
parser.add_argument('--save-stats', default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument(
    '--amp-theta', default=[0, 12, 3, -4, -8, 2], type=list, help='theta reference state in degrees.')
parser.add_argument('--amp-phi', default=[2, -2, 2, 10, 2, -6], type=list, help='phi reference state in degrees.')
parser.add_argument('--compare-performance',
                    default=False, action='store_true', help='compare performance of an agent between environments')
parser.add_argument('--filter-action', action='store_true', default=False)
############################################

# parsed_param, unknown = parser.parse_known_args()
parsed_param = parser.parse_args()
parsed_param.max_theta = max(np.abs(parsed_param.amp_theta))
parsed_param.max_phi = max(np.abs(parsed_param.amp_phi))

params = ESParameters(conf=parsed_param, init=True)

ENVS = dict(
    LUNAR_LANDER='LunarLanderContinuous-v2',  # for quick tests
    NOMINAL='PHlab_attitude_nominal',
    ICE='PHlab_attitude_ice',
    CG_SHIFT='PHlab_attitude_cg-shift',  # cg shift aft after 20s
    SATURATED_AILERON='PHlab_attitude_sa',
    SATURATED_ELEVATOR='PHlab_attitude_se',
    BROKEN_ELEVATOR='PHlab_attitude_be',
    JAMMED_RUDDER='PHlab_attitude_jr',
    CG_FOR='PHlab_attitude_cg-for',
    CG_AFT='PHlab_attitude_cg',

    HIGH_Q='PHlab_attitude_high-q',
    LOW_Q='PHlab_attitude_low-q',
    NOISE='PHlab_attitude_noise',
    GUST='PHlab_attitude_gust',
)

# print(f'Using Mu: {parsed_param.use_mu}')
t_max = parsed_param.t_max

env = select_env(
    parsed_param.env_name,
    render_mode=False,
    realtime=False,
    use_state_history=parsed_param.use_state_history,
    conform_with_sb=True
)

env.set_eval_mode(t_max=t_max)

# ****** Adding a fault ******
if parsed_param.use_second_env:
    env2 = select_env(
        parsed_param.env_name,
        render_mode=False,
        realtime=False,
        use_state_history=parsed_param.use_state_history,
        conform_with_sb=True
    )
    env2.set_eval_mode(t_max=t_max)
    total_timesteps = int(parsed_param.t_max//0.01)
    switch_timestep = int(parsed_param.switch_time//0.01)
    remaining_timesteps = int(total_timesteps-switch_timestep)

elif parsed_param.compare_performance:
    env2 = select_env(
        parsed_param.env2_name,
        render_mode=False,
        realtime=False,
        use_state_history=parsed_param.use_state_history,
        conform_with_sb=True
    )
    env2.set_eval_mode(t_max=t_max)
    envs = [env, env2]


def eval_cemtd3(actor, **kwargs):
    mult_data = []
    mult_sm = []
    mult_nmae = []
    for env in envs:
        x_ctrl_lst = []
        errors = []
        ref_lst = []
        t_lst = []

        state_lst, rewards, action_lst = [], [], []

        obs, _ = env.reset(**kwargs)
        for s in range(8001):
            state_lst.append(env.x)
            action_lst.append(env.last_u)
            x_ctrl_lst.append(env.get_controlled_state())

            # select the action:
            action = actor.select_action(obs)
            action = np.clip(action, -1, 1)
            ref_value = np.deg2rad(
                np.array([ref(env.t) for ref in env.ref]).flatten())
            next_obs, reward, done, _, info = env.step(action.flatten())

            obs = next_obs

            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env.t)

        errors = np.asarray(errors)
        nmae = calc_nMAE(errors)
        mult_nmae.append(nmae)
        actions = np.array(action_lst)
        sm = calc_smoothness(actions, plot_spectra=False)
        mult_sm.append(sm)
        ref_values = np.array(ref_lst)
        t_lst = np.asarray(t_lst).reshape((-1, 1))
        data = np.concatenate(
            (ref_values, actions, state_lst, t_lst), axis=1)
        mult_data.append(data)

    mult_data = np.asarray(mult_data)

    return mult_data, mult_nmae, mult_sm


def evaluate_trained_agent(actor, **kwargs):
    """Simulate one episode in the environment with the given actor"""

    # total number of steps is at most t_max//dt:
    x_lst, rewards, u_lst = [], [], []
    x_ctrl_lst = []
    errors = []
    ref_lst = []
    t_lst = []

    # reset the environment:
    obs, _ = env.reset(**kwargs)
    if parsed_param.use_second_env:
        env2.reset(**kwargs)
        for s in range(total_timesteps):
            x_lst.append(env.x)
            u_lst.append(env.last_u)
            x_ctrl_lst.append(env.get_controlled_state())

            # select the action:
            action = actor.select_action(obs)

            # simulate one step into the env:
            # clip the action to -1, 1
            action = np.clip(action, -1, 1)
            ref_value = np.deg2rad(
                np.array([ref(env.t) for ref in env.ref]).flatten())
            obs, reward, done, _, info = env.step(action.flatten())
            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env.t)

            if s == switch_timestep:
                env2.t = deepcopy(env.t)
                env2.x = deepcopy(env.x)
                env2.obs = deepcopy(obs)
                env2.last_u = deepcopy(env.last_u)
                break

        for s in range(remaining_timesteps):
            x_lst.append(env2.x)
            u_lst.append(env2.last_u)
            x_ctrl_lst.append(env2.get_controlled_state())

            # select the action:
            action = actor.select_action(env2.obs)

            # simulate one step into the env:
            # clip the action to -1, 1
            action = np.clip(action, -1, 1)
            ref_value = np.deg2rad(
                np.array([ref(env2.t) for ref in env2.ref]).flatten())
            obs, reward, done, _, info = env2.step(action.flatten())
            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env2.t)

        env2.close()
        env.close()
    else:
        done = False
        while not done:
            x_lst.append(env.x)
            u_lst.append(env.last_u)
            x_ctrl_lst.append(env.get_controlled_state())

            # select the action:
            action = actor.select_action(obs)

            # simulate one step into the env:
            # clip the action to -1, 1
            action = np.clip(action, -1, 1)
            ref_value = np.deg2rad(
                np.array([ref(env.t) for ref in env.ref]).flatten())
            next_obs, reward, done, _, info = env.step(action.flatten())

            if kwargs.get('verbose'):
                print(
                    f'Action: {np.rad2deg(action)} -> deflection: {np.rad2deg(env.last_u)}')
                print(
                    f't:{env.t:0.2f} theta:{env.theta:.03f} q:{env.q:.03f} phi:{env.phi:.03f} alpha:{env.alpha:.03f} V:{env.V:.03f} H:{env.h:.03f}')
                print(
                    f'Error: {np.rad2deg(obs[:env.n_actions])} Reward: {reward:.03f} \n\n')

            # Update:
            obs = next_obs

            # save:
            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env.t)

    env.finish()

    # control inputs
    errors = np.asarray(errors)

    # compute the scaled smoothness fitness:
    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions, **kwargs)

    # format data:
    rewards = np.asarray(rewards).reshape((-1, 1))
    ref_values = np.array(ref_lst)
    t_lst = np.asarray(t_lst).reshape((-1, 1))
    data = np.concatenate((ref_values, actions, x_lst, rewards, t_lst), axis=1)

    # calculate nMAE:
    nmae = calc_nMAE(errors)

    return data, nmae, smoothness


def validate_trained_agent(gen_agent, user_ref_lst: list, num_trials: int = 1, **kwargs):
    """ Evaluate the agent over the given reference signals and number of trials
    Note that the time traces from data come from the LAST EPISODE
    """
    agent_nmae_lst, agent_sm_lst = [], []

    for i in tqdm(range(num_trials+1), total=num_trials):
        ref_t = user_ref_lst[i]
        user_refs = {
            'theta_ref': ref_t[0],
            'phi_ref': ref_t[1],
            # 'psi_ref': ref_t[2],
        }
        if parsed_param.compare_performance:
            data, nmae, smoothness = eval_cemtd3(
                gen_agent, user_refs=user_refs, **kwargs)
        else:
            data, nmae, smoothness = evaluate_trained_agent(
                gen_agent, user_refs=user_refs, **kwargs)
        agent_nmae_lst.append(nmae)
        agent_sm_lst.append(smoothness)

    if parsed_param.compare_performance:
        agent_nmae_lst = np.asarray(agent_nmae_lst)
        agent_sm_lst = np.asarray(agent_sm_lst)
        nmae = np.round(np.mean(agent_nmae_lst, axis=0), 2)
        nmae_std = np.std(agent_nmae_lst, axis=0)
        smoothness = np.round(np.median(agent_sm_lst, axis=0), 2)
        sm_std = np.std(agent_sm_lst, axis=0)
    else:
        nmae = np.mean(agent_nmae_lst)
        nmae_std = np.std(agent_nmae_lst)
        smoothness = np.median(agent_sm_lst)
        sm_std = np.std(agent_sm_lst)

    # if kwargs.get('stdout'):
    #     print(f'nMAE: {nmae:0.1f}% with STD: {nmae_std:0.1f}')
    #     print(f'Smoothness: {smoothness:0.0f} with STD: {sm_std:0.1f}')
    if parsed_param.compare_performance:
        return data, nmae, smoothness
    else:
        stats = Stats(nmae, nmae_std, smoothness, sm_std)
        return data, stats


def main():
    # seed:
    # env.seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)

    params.action_dim = env.action_space.shape[0]
    params.state_dim = env.observation_space.shape[0]

    # TODO: Identify the fault:
    _, _, fault_name = parsed_param.env_name.split('_')

    ''' Prepare References '''
    # Time set points:
    time_array = np.linspace(0., t_max, 6)

    # pitch set-points:
    amp1_n = [0, 12, 3, -4, 4, 1]
    amp1_n2 = [0, -2, 3, 5, -1, 2]
    amp1 = [0, 12, 3, -4, -8, 2]
    base_ref_theta = SmoothedStepSequence(
        times=time_array, amplitudes=parsed_param.amp_theta, smooth_width=t_max//10)

    # Roll set-points:
    amp2_n = [0, 0, 0, 0, 0, 0]
    amp2_n2 = [1, -2, 2, -3, 4, -1]
    amp2 = [2, -2, 2, 10, 2, -6]
    base_ref_phi = SmoothedStepSequence(
        times=time_array, amplitudes=parsed_param.amp_phi, smooth_width=t_max//10)

    # build list of reference tuples from seed:
    theta_refs = gen_refs(t_max=t_max, ampl_times=time_array,
                          ampl_max=parsed_param.max_theta, num_trials=parsed_param.num_trials)

    phi_refs = gen_refs(t_max, time_array, ampl_max=parsed_param.max_phi,
                        num_trials=parsed_param.num_trials)
    theta_refs.append(base_ref_theta)
    phi_refs.append(base_ref_phi)

    user_eval_refs = list(zip(theta_refs, phi_refs))

    # Load agent info
    logs_dir = find_logs_path(parsed_param.agent_name)
    print(logs_dir)
    model_config = load_config(logs_dir)

    # update config:
    params.update_from_dict(model_config)
    setattr(params, 'env-name', params.env_name)
    params.stdout()

    # build path to figures:
    fig_path = logs_dir / Path('figures')
    fault_path = fig_path / Path(fault_name)

    '''   Load Champion Actor '''
    actor_params = deepcopy(params)
    actor_params.update_from_dict(model_config)

    agent = load_agent(logs_dir, actor_params, parsed_param)
    # if parsed_param.use_mu:
    #     print('Using Mu actor')
    #     agent = load_mu_agent(logs_dir, args=actor_params)
    # else:
    #     agent = load_agent(
    #         logs_dir, args=actor_params)
    if parsed_param.compare_performance:
        data, nmae, smoothness = validate_trained_agent(
            agent, user_eval_refs, num_trials=parsed_param.num_trials, stdout=True, plot_spectra=parsed_param.plot_spectra)
    else:
        data, stats = validate_trained_agent(
            agent, user_eval_refs, num_trials=parsed_param.num_trials, stdout=True, plot_spectra=parsed_param.plot_spectra)

    print(data.shape)
    # ---- fig name prefix:
    if parsed_param.use_mu:
        figname_prefix = 'mu_'
    elif parsed_param.use_best_mu:
        figname_prefix = 'best_mu_'
    elif parsed_param.use_best_elite:
        figname_prefix = 'best_elite_'
    else:
        figname_prefix = 'elite_'

    faults = {
        'nominal': 'Normal',
        'ice': 'Iced Wing',
        'cg-shift': 'Shifted CG',
        'sa': 'Saturated Aileron',
        'se': 'Saturated Elevator',
        'be': 'Partial Loss of Elevator',
        'jr': 'Jammed Rudder',
        'high-q': 'High Q',
        'low-q': 'Low Q',
        'noise': 'Sensor Noise',
        'gust': 'Gust of Wind',
        'cg-for': 'Forward Shifted CG',
        'cg': 'Backward Shifted CG',
    }

    if parsed_param.use_second_env:
        switch_time = parsed_param.switch_time
        _, _, f1 = parsed_param.env_name.split('_')
        _, _, f2 = parsed_param.env2_name.split('_')
        faultName = f"{faults[f1]}_to_{faults[f2]}_at_{switch_time}_sec"
        figname = f"{faultName}_{parsed_param.agent_name}__{figname_prefix}"
    elif parsed_param.compare_performance:
        _, _, f1 = parsed_param.env_name.split('_')
        _, _, f2 = parsed_param.env2_name.split('_')
        faultname = f"[nMAE {nmae[0]}%|Sm {smoothness[0]} Rad.Hz] vs [nMAE {nmae[1]}%|Sm {smoothness[1]} Rad.Hz]"
        name = f"{faults[f1]}_vs_{faults[f2]}"
        figname = f"{name}_{parsed_param.agent_name}__{figname_prefix}"
    else:
        faultName = faults[fault_name]
        figname = f"{faultName}_{parsed_param.agent_name}__{figname_prefix}"
        switch_time = None
    # plot the actor result:

    if parsed_param.compare_performance:
        fig = plot_comparative_analysis(data, faultname, fig_name=figname)
    else:
        fig, _ = plot(
            data, name=f'nMAE: {stats.nmae:0.2f}% - Smoothness: {stats.sm:0.2f} rad.Hz', fault=faultName, fig_name=figname, switch_time=switch_time, filter_action=parsed_param.filter_action,
        )

    # if parsed_param.save_plots:
    #     if not os.path.exists(fig_path):
    #         os.mkdir(fig_path)
    #     if not os.path.exists(fault_path):
    #         os.mkdir(fault_path)
    #     fig_name = fault_path / \
    #         Path(figname_prefix + fault_name + '.png')
    #     # fig_name = fault_path / \
    #     #     Path('elite_' + fault_name + '.png') if not parsed_param.use_mu else fault_path / \
    #     #     Path('mu_' + fault_name + '.png')
    #     fig.savefig(fname=fig_name, dpi=300, format='png')
    #     plt.close()
    # else:
    #     plt.show()

    '''   Save data files  '''
    if parsed_param.save_stats:
        toml_path = logs_dir / Path(figname_prefix + 'stats.toml')
        stats_dict = {fault_name: stats.__dict__}

        with open(toml_path, 'a+', encoding='utf-8') as f:
            f.write('\n\n')
            toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

        f.close()

    # Save trajectories on base reference:
    if parsed_param.save_trajectory:
        save_trajectory(fault_path, data, figname_prefix)


def save_trajectory(fault_path, data, agent_type):
    """save the actor trajectory data

    Args:
        fault_path (_type_): _description_
        data (_type_): _description_
    """
    save_path = fault_path / Path(agent_type + 'trajectory.csv')
    with open(save_path, 'w+', encoding='utf-8') as fp:
        np.savetxt(fp, data)

    print(f'Trajectory saved as: {save_path}')
    fp.close()


if __name__ == '__main__':
    main()
