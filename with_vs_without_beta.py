import copy
import os
from pathlib import Path
from matplotlib import font_manager, pyplot as plt
import toml
from tqdm import tqdm
from core_algorithms.utils import Episode, calc_nMAE, calc_smoothness, load_config
from evaluation_es_utils import gen_eval_refs
from parameters_es import ESParameters
from plotters.plot_utils import plot, plot_attitude_response
from ppo_continuous_actions import Agent, make_env
from environments.config import select_env
import torch
import gymnasium as gym
import argparse

import numpy as np


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


parser = argparse.ArgumentParser(description='PPO')
parser.add_argument(
    '--gym-id', default='PHlab_attitude_nominal', type=str, help='environment')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num-trials', default=2, type=int)
parser.add_argument('--t-max', type=int, default=80)
parser.add_argument('--add-sm-to-reward',
                    action='store_true', default=False)
parser.add_argument('--use-scaled-obs', action='store_true', default=False)
parser.add_argument('--num-envs', default=1, type=int, help='number of envs')
parser.add_argument('--seed', default=7, type=int, help='seed')
# parser.add_argument('--fdd', type=str, default='none')
# [0, 12, 3, -4, -8, 2], [0, 20, 10, 1, 0, -2]
parser.add_argument(
    '--amp-theta', default=[0, 12, 3, -4, -8, 2], type=list_of_ints)

# [2, -2, 2, 10, 2, -6], [0, 0, 0, 40, 2, -6]
parser.add_argument(
    '--amp-phi', default=[2, -2, 2, 10, 2, -6], type=list_of_ints)
parser.add_argument('--filter-action', action='store_true', default=False)


def plot_with_vs_without_beta(data_beta, data_withoutBeta, faultname, fig_name=None):
    import matplotlib as mplt

    fontpath = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    mplt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.rcParams['lines.markersize'] = 4

    def filter_outliers(data, t):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filter_idx = ((data > lower_bound) & (data < upper_bound))
        return t[filter_idx], data[filter_idx]

    fontsize = 14
    labels = ['Tracked States', 'Reference', 'Actuator Deflection']
    ref_values = data_beta[:, :3]
    action_beta = data_beta[:, 3:6]
    action_withoutBeta = data_withoutBeta[:, 2:5]

    x_lst_beta = data_beta[:, 6:16]
    x_lst_withoutBeta = data_withoutBeta[:, 5:15]
    _t = data_beta[:, -1]

    theta_ref, phi_ref, psi_ref = ref_values[:, 0], ref_values[:, 1], \
        ref_values[:, 2]
    theta_ref = np.rad2deg(theta_ref)
    phi_ref = np.rad2deg(phi_ref)
    psi_ref = np.rad2deg(psi_ref)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(_t, theta_ref, linestyle='--',
                   color='black', label=labels[1], linewidth=3)
    axs[0, 0].set_ylabel(r'$\theta \:\: [{deg}]$', fontsize=fontsize)
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].plot(_t, phi_ref, linestyle='--', color='black', linewidth=3)
    axs[1, 0].set_ylabel(r'$\phi \:\: [{deg}]$', fontsize=fontsize)
    axs[1, 0].grid()
    axs[1, 1].grid()
    axs[2, 0].plot(_t, psi_ref, linestyle='--', color='black',
                   linewidth=3)
    axs[2, 0].set_ylabel(r'$\beta \:\: [{deg}]$', fontsize=fontsize)
    axs[2, 0].grid()
    axs[2, 1].grid()
    axs[2, 0].set_xlabel('Time [S]', fontsize=fontsize)
    axs[2, 1].set_xlabel('Time [S]', fontsize=fontsize)
    tracked_state_color = 'magenta'
    action_color = 'green'
    # i = 0
    line_styles = ['-', 'dotted']
    x_lsts = [x_lst_beta, x_lst_withoutBeta]
    actions = [action_beta, action_withoutBeta]
    for i in range(len(line_styles)):
        p, q, r, V, alpha, beta, phi, theta, psi, h = x_lsts[i][:, 0], x_lsts[i][:, 1], \
            x_lsts[i][:, 2], x_lsts[i][:, 3], x_lsts[i][:, 4], x_lsts[i][:, 5], x_lsts[i][:, 6], x_lsts[i][:, 7], \
            x_lsts[i][:, 8], x_lsts[i][:, 9]

        de, da, dr = actions[i][:, 0], actions[i][:, 1], actions[i][:, 2]
        de = np.rad2deg(de)
        da = np.rad2deg(da)
        dr = np.rad2deg(dr)
        t_de, de_filt = filter_outliers(de, _t)
        t_da, da_filt = filter_outliers(da, _t)
        t_dr, dr_filt = filter_outliers(dr, _t)

        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
        psi = np.rad2deg(psi)
        p = np.rad2deg(p)
        q = np.rad2deg(q)
        r = np.rad2deg(r)
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)

        axs[0, 0].plot(_t, theta,
                       linestyle=line_styles[i], color=tracked_state_color, label=labels[0], linewidth=2)
        axs[0, 1].plot(t_de, de_filt, label=labels[2], color=action_color,
                       linestyle=line_styles[i], linewidth=2)
        axs[0, 1].set_ylabel(r'$\delta_e \:\: [{deg}]$', fontsize=fontsize)

        axs[1, 0].plot(_t, phi,
                       linestyle=line_styles[i],
                       color=tracked_state_color, linewidth=2)
        axs[1, 1].plot(t_da, da_filt, color=action_color,
                       linestyle=line_styles[i], linewidth=2)
        axs[1, 1].set_ylabel(r'$\delta_a \:\: [{deg}]$', fontsize=fontsize)

        axs[2, 0].plot(_t, beta,
                       linestyle=line_styles[i], color=tracked_state_color, linewidth=2)
        axs[2, 1].plot(t_dr, dr_filt, color=action_color,
                       linestyle=line_styles[i], linewidth=2)
        axs[2, 1].set_ylabel(r'$\delta_r \:\: [{deg}]$', fontsize=fontsize)

    leg_lines = []
    leg_labels = []
    for ax in fig.axes:
        axLine, axLegend = ax.get_legend_handles_labels()
        leg_lines.extend(axLine)
        leg_labels.extend(axLegend)

    fig.legend(leg_lines, leg_labels, loc='upper center',
               ncol=5, mode='expand', bbox_to_anchor=(0.11, 0.68, 0.8, 0.25), fontsize=13)
    fig.suptitle(faultname, fontsize=fontsize)

    if fig_name is not None:
        fig_path = Path(os.getcwd()) / Path('figures_withoutSuptitle')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        figname = fig_path / Path(fig_name + '.pdf')
        # fig.write_image(str(figname))
        fig.savefig(figname, dpi=300, bbox_inches='tight', format='pdf')
    # plt.show()
    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    args.max_theta = max(np.abs(args.amp_theta))
    args.max_phi = max(np.abs(args.amp_phi))

    run_name_beta = 'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl'
    run_name_withoutBeta = 'PHlab_attitude_nominal__ppo_continous_actions__1__1711972191_addsmtorw_maxdiffsm_False__multipleEnvs_False_singleENV__totalTimesteps_2000000__useScaledObs_False__NoBeta.pkl'

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # ****** Define Env ******#
    envs_beta = gym.vector.SyncVectorEnv(
        [make_env(gym_id=args.gym_id, seed=args.seed+i, idx=i, capture_video=False, run_name=run_name_beta, eval=True, use_scaled_obs=args.use_scaled_obs) for i in range(args.num_envs)])
    envs_withoutBeta = gym.vector.SyncVectorEnv(
        [make_env(gym_id=args.gym_id, seed=args.seed+i, idx=i, capture_video=False, run_name=run_name_withoutBeta, eval=True, use_scaled_obs=args.use_scaled_obs, without_beta=True) for i in range(args.num_envs)])

    # load agents:
    agent_beta = Agent(envs=envs_beta, eval=True).to(device)
    path = 'agents/' + run_name_beta
    agent_beta.load_state_dict(torch.load(path))
    print("Agent loaded from: ", path)

    agent_withoutBeta = Agent(envs=envs_withoutBeta, eval=True).to(device)
    path = 'agents/' + run_name_withoutBeta
    agent_withoutBeta.load_state_dict(torch.load(path))
    print("Agent loaded from: ", path)

    # ****** Define Evaluation ******#
    user_eval_refs = gen_eval_refs(
        amp_theta=args.amp_theta,
        amp_phi=args.amp_phi,
        t_max=args.t_max,
        max_theta=args.max_theta,
        max_phi=args.max_phi,
        num_trails=args.num_trials
    )

    env_beta = select_env(
        environment_name=args.gym_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward,
        use_scaled_obs=args.use_scaled_obs,
    )
    env_beta.set_eval_mode(t_max=args.t_max)
    t_max = args.t_max
    # if args.use_second_env:
    env_withoutBeta = select_env(
        environment_name=args.gym_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward,
        use_scaled_obs=args.use_scaled_obs,
        without_beta=True
    )
    env_withoutBeta.set_eval_mode(t_max=args.t_max)

    # ****** Evaluate ******#
    def eval_agents(**kwargs):
        mult_data = []
        mult_sm = []
        mult_nmae = []

        # ********* with beta ********** #
        x_ctrl_lst = []
        errors = []
        ref_lst = []
        t_lst = []

        state_lst, rewards, action_lst = [], [], []

        obs, _ = env_beta.reset(**kwargs)

        for s in range(int(args.t_max/0.01)):
            action = agent_beta.get_action_and_value(
                torch.FloatTensor(obs))[-1].detach().numpy()
            ref_value = np.deg2rad(
                np.array([ref(env_beta.t) for ref in env_beta.ref]).flatten())
            obs, reward, done, truncated, info = env_beta.step(action)
            action_lst.append(env_beta.last_u)
            state_lst.append(env_beta.x)
            x_ctrl_lst.append(env_beta.get_controlled_state())

            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env_beta.t)

        errors = np.asarray(errors)
        nmae_beta = calc_nMAE(errors)
        # mult_nmae.append(nmae)
        actions = np.array(action_lst)
        sm_beta = calc_smoothness(actions, plot_spectra=False)
        # mult_sm.append(sm)
        ref_values = np.array(ref_lst)
        t_lst = np.asarray(t_lst).reshape((-1, 1))
        data_beta = np.concatenate(
            (ref_values, actions, state_lst, t_lst), axis=1)
        # mult_data.append(data)

        # mult_data = np.concatenate(mult_data, axis=0)
        # mult_data = np.asarray(mult_data)

        # ********* without beta ********** #
        x_ctrl_lst = []
        errors = []
        ref_lst = []
        t_lst = []

        state_lst, rewards, action_lst = [], [], []

        obs, _ = env_withoutBeta.reset(**kwargs)

        for s in range(int(args.t_max/0.01)):
            action = agent_withoutBeta.get_action_and_value(
                torch.FloatTensor(obs))[-1].detach().numpy()
            ref_value = np.deg2rad(
                np.array([ref(env_withoutBeta.t) for ref in env_withoutBeta.ref]).flatten())
            obs, reward, done, truncated, info = env_withoutBeta.step(action)
            action_lst.append(env_withoutBeta.last_u)
            state_lst.append(env_withoutBeta.x)
            x_ctrl_lst.append(env_withoutBeta.get_controlled_state())

            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env_withoutBeta.t)

        errors = np.asarray(errors)
        nmae_withoutBeta = calc_nMAE(errors)
        # mult_nmae.append(nmae)
        actions = np.array(action_lst)
        sm_wihoutBeta = calc_smoothness(actions, plot_spectra=False)
        # mult_sm.append(sm)
        ref_values = np.array(ref_lst)
        t_lst = np.asarray(t_lst).reshape((-1, 1))
        data_withoutBeta = np.concatenate(
            (ref_values, actions, state_lst, t_lst), axis=1)
        # mult_data.append(data)

        return data_beta, nmae_beta, sm_beta, data_withoutBeta, nmae_withoutBeta, sm_wihoutBeta

    agent_nmae_lst_beta, agent_nmae_lst_withoutBeta, agent_sm_lst_beta, agent_sm_lst_withoutBeta = [], [], [], []
    for i in tqdm(range(args.num_trials+1), total=args.num_trials):
        ref_t = user_eval_refs[i]
        user_refs = {
            'theta_ref': ref_t[0],
            'phi_ref': ref_t[1]
        }

        data_beta, nmae_beta, sm_beta, data_withoutBeta, nmae_withoutBeta, sm_withoutBeta = eval_agents(
            user_refs=user_refs)
        agent_nmae_lst_beta.append(nmae_beta)
        agent_nmae_lst_withoutBeta.append(nmae_withoutBeta)
        agent_sm_lst_beta.append(sm_beta)
        agent_sm_lst_withoutBeta.append(sm_withoutBeta)
        # print(sm)

    agent_nmae_lst_beta = np.asarray(agent_nmae_lst_beta)
    agent_sm_lst_beta = np.asarray(agent_sm_lst_beta)
    nmae_beta = np.round(np.mean(agent_nmae_lst_beta, axis=0), 2)
    nmae_std_beta = np.std(agent_nmae_lst_beta, axis=0)
    smoothness_beta = np.round(np.median(agent_sm_lst_beta, axis=0), 2)
    sm_std_beta = np.std(agent_sm_lst_beta, axis=0)

    agent_nmae_lst_withoutBeta = np.asarray(agent_nmae_lst_withoutBeta)
    agent_sm_lst_withoutBeta = np.asarray(agent_sm_lst_withoutBeta)
    nmae_withoutBeta = np.round(np.mean(agent_nmae_lst_withoutBeta, axis=0), 2)
    nmae_std_withoutBeta = np.std(agent_nmae_lst_withoutBeta, axis=0)
    smoothness_withoutBeta = np.round(
        np.median(agent_sm_lst_withoutBeta, axis=0), 2)
    sm_std_withoutBeta = np.std(agent_sm_lst_withoutBeta, axis=0)

    faults = {
        'nominal': 'Normal',
        'ice': 'Iced Wings',
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
    _, _, f1 = args.gym_id.split('_')

    faultname = f"With Beta [nMAE {nmae_beta}%|Sm {smoothness_beta} Rad.Hz] vs Without Beta [nMAE {nmae_withoutBeta}%|Sm {smoothness_withoutBeta} Rad.Hz]"

    name = f"{faults[f1]}_with_or_without_beta"
    figname = f"{name}_ppo"

    fig = plot_with_vs_without_beta(
        data_beta, data_withoutBeta, faultname, figname)
