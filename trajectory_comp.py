import os
from pathlib import Path
from matplotlib import font_manager, pyplot as plt
import toml
from tqdm import tqdm
from core_algorithms.utils import calc_nMAE, calc_smoothness
from evaluation_es_utils import Stats, gen_eval_refs
# from parameters_es import ESParameters
# from plotters.plot_utils import plot, plot_attitude_response
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

# [0, 12, 3, -4, -8, 2], [0, 20, 10, 1, 0, -2]
parser.add_argument(
    '--amp-theta1', default=[0, 12, 3, -4, -8, 2], type=list_of_ints)
parser.add_argument(
    '--amp-theta2', default=[0, 20, 10, 1, 0, -2], type=list_of_ints)
parser.add_argument(
    '--amp-theta3', default=[0, -5, 10, 5, 0, 0], type=list_of_ints)

# [2, -2, 2, 10, 2, -6], [0, 0, 0, 40, 2, -6]
parser.add_argument(
    '--amp-phi1', default=[2, -2, 2, 10, 2, -6], type=list_of_ints)
parser.add_argument(
    '--amp-phi2', default=[0, 0, 0, 40, 2, -6], type=list_of_ints)
parser.add_argument(
    '--amp-phi3', default=[-5, 2, -3, 0, 10, -10], type=list_of_ints)
parser.add_argument('--filter-action', action='store_true', default=False)
parser.add_argument('--agent-name', default='PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl', help='PPO trained agent', required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    # trajectories preparation:
    args.max_theta1 = max(np.abs(args.amp_theta1))
    args.max_phi1 = max(np.abs(args.amp_phi1))
    args.max_theta2 = max(np.abs(args.amp_theta2))
    args.max_phi2 = max(np.abs(args.amp_phi2))
    args.max_theta3 = max(np.abs(args.amp_theta3))
    args.max_phi3 = max(np.abs(args.amp_phi3))

    user_eval_refs_1 = gen_eval_refs(
        amp_theta=args.amp_theta1,
        amp_phi=args.amp_phi1,
        t_max=args.t_max,
        max_theta=args.max_theta1,
        max_phi=args.max_phi1,
        num_trails=args.num_trials
    )
    user_eval_refs_2 = gen_eval_refs(
        amp_theta=args.amp_theta2,
        amp_phi=args.amp_phi2,
        t_max=args.t_max,
        max_theta=args.max_theta2,
        max_phi=args.max_phi2,
        num_trails=args.num_trials
    )
    user_eval_refs_3 = gen_eval_refs(
        amp_theta=args.amp_theta3,
        amp_phi=args.amp_phi3,
        t_max=args.t_max,
        max_theta=args.max_theta3,
        max_phi=args.max_phi3,
        num_trails=args.num_trials
    )

    user_eval_refs = [user_eval_refs_1, user_eval_refs_2, user_eval_refs_3]

    # load agent:
    run_name = 'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl'

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id=args.gym_id, seed=args.seed+i, idx=i, capture_video=False, run_name=run_name, eval=True, use_scaled_obs=args.use_scaled_obs) for i in range(args.num_envs)])

    agent = Agent(envs=envs, eval=True).to(device)
    path = 'agents/' + run_name
    agent.load_state_dict(torch.load(path))

    print("Agent loaded from: ", path)

    # ******* Define Environment *********
    env = select_env(
        environment_name=args.gym_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward,
        use_scaled_obs=args.use_scaled_obs,
    )
    env.set_eval_mode(t_max=args.t_max)

    def eval_multiple_trajectories(**kwargs):
        x_ctrl_lst = []
        errors = []
        ref_lst = []
        t_lst = []

        state_lst, rewards, action_lst = [], [], []

        obs, _ = env.reset(**kwargs)

        for s in range(8001):
            action = agent.get_action_and_value(
                torch.FloatTensor(obs))[-1].detach().numpy()
            ref_value = np.deg2rad(
                np.array([ref(env.t) for ref in env.ref]).flatten())
            obs, reward, done, truncated, info = env.step(action)
            action_lst.append(env.last_u)
            state_lst.append(env.x)
            x_ctrl_lst.append(env.get_controlled_state())

            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env.t)

        errors = np.asarray(errors)
        nmae = calc_nMAE(errors)
        actions = np.array(action_lst)
        sm = calc_smoothness(actions, plot_spectra=False)
        ref_values = np.array(ref_lst)
        t_lst = np.asarray(t_lst).reshape((-1, 1))
        data = np.concatenate(
            (ref_values, actions, state_lst, t_lst), axis=1)

        return data, nmae, sm

    all_data, all_nmae, all_sm = [], [], []
    for user_eval_ref in user_eval_refs:
        traj_nmae, traj_sm = [], []
        for i in tqdm(range(args.num_trials+1), total=args.num_trials+1):
            ref_t = user_eval_ref[i]
            user_refs = {
                'theta_ref': ref_t[0],
                'phi_ref': ref_t[1],
            }

            data, nmae, sm = eval_multiple_trajectories(
                user_refs=user_refs)
            traj_nmae.append(nmae)
            traj_sm.append(sm)

        all_data.append(data)
        all_nmae.append(np.round(np.mean(traj_nmae), 2))
        all_sm.append(np.round(np.median(traj_sm), 2))
    all_data = np.array(all_data)
    all_nmae = np.array(all_nmae)
    all_sm = np.array(all_sm)

    print(all_data.shape)
    print(all_nmae)
    print(all_sm)

    faults = {
        'nominal': 'Normal Flight',
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

    _, _, f = args.gym_id.split('_')
    faultname = f"[AvgnMAE {np.round(np.mean(all_nmae), 2)}% |AvgSm {np.round(np.mean(all_sm), 2)} Rad.Hz]"
    name = f"{faults[f]}_traj_comparison"
    figname = f"{name}_{run_name.split('.')[0]}"

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

    # ref_values
    ref_values1 = all_data[0, :, :3]
    ref_values2 = all_data[1, :, :3]
    ref_values3 = all_data[2, :, :3]
    theta_ref1, phi_ref1, psi_ref1 = np.rad2deg(
        ref_values1[:, 0]), np.rad2deg(ref_values1[:, 1]), np.rad2deg(ref_values1[:, 2])
    theta_ref2, phi_ref2, psi_ref2 = np.rad2deg(
        ref_values2[:, 0]), np.rad2deg(ref_values2[:, 1]), np.rad2deg(ref_values2[:, 2])
    theta_ref3, phi_ref3, psi_ref3 = np.rad2deg(ref_values3[:, 0]), np.rad2deg(
        ref_values3[:, 1]), np.rad2deg(ref_values3[:, 2])

    theta_refs = [theta_ref1, theta_ref2, theta_ref3]
    phi_refs = [phi_ref1, phi_ref2, phi_ref3]
    psi_refs = [psi_ref1, psi_ref2, psi_ref3]

    # actions:
    actions1 = all_data[0, :, 3:6]
    actions2 = all_data[1, :, 3:6]
    actions3 = all_data[2, :, 3:6]

    actions = [actions1, actions2, actions3]
    # states:
    x_lst1 = all_data[0, :, 6:16]
    x_lst2 = all_data[1, :, 6:16]
    x_lst3 = all_data[2, :, 6:16]
    x_lsts = [x_lst1, x_lst2, x_lst3]

    _t = all_data[0, :, -1]

    fig, axs = plt.subplots(3, 2)
    # 'loosely dotted',        (0, (1, 10))
    # 'loosely dashed',        (0, (5, 10))
    # 'loosely dashdotted',    (0, (3, 10, 1, 10))
    # (0, (3, 5, 1, 5, 1, 5))
    # ref_line_styles = [(0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10))]
    ref_line_styles = ['dashed', 'dotted', 'dashdot']

    # action_line_styles = ['-', '--', '-.']
    ref_line_colors = ['magenta', 'olive', 'cadetblue']

    state_line_style = '-'
    tracked_state_colors = ['magenta', 'olive', 'cadetblue']
    action_colors = ['magenta', 'olive', 'cadetblue']

    for i in range(len(ref_line_styles)):
        # plot references
        axs[0, 0].plot(_t, theta_refs[i], linestyle=ref_line_styles[i],
                       color=ref_line_colors[i], linewidth=3, label=labels[1])
        axs[0, 0].set_ylabel(r'$\theta \:\: [{deg}]$', fontsize=fontsize)
        axs[0, 0].grid()
        axs[0, 1].grid()
        axs[1, 0].plot(_t, phi_refs[i], linestyle=ref_line_styles[i],
                       color=ref_line_colors[i], linewidth=3)
        axs[1, 0].set_ylabel(r'$\phi \:\: [{deg}]$', fontsize=fontsize)
        axs[1, 0].grid()
        axs[1, 1].grid()
        axs[2, 0].plot(_t, psi_refs[i], linestyle=ref_line_styles[i],
                       color=ref_line_colors[i], linewidth=3)
        axs[2, 0].set_ylabel(r'$\beta \:\: [{deg}]$', fontsize=fontsize)
        axs[2, 0].grid()
        axs[2, 1].grid()

        # states:
        beta, phi, theta = x_lsts[i][:, 5], x_lsts[i][:, 6], x_lsts[i][:, 7]
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
        beta = np.rad2deg(beta)
        axs[0, 0].plot(_t, theta,
                       linestyle=state_line_style, color=tracked_state_colors[i], label=labels[0], linewidth=2)
        axs[1, 0].plot(_t, phi,
                       linestyle=state_line_style,
                       color=tracked_state_colors[i], linewidth=2)
        axs[2, 0].plot(_t, beta,
                       linestyle=state_line_style, color=tracked_state_colors[i], linewidth=2)

        # actions
        de, da, dr = actions[i][:, 0], actions[i][:, 1], actions[i][:, 2]
        de = np.rad2deg(de)
        da = np.rad2deg(da)
        dr = np.rad2deg(dr)
        t_de, de_filt = filter_outliers(de, _t)
        t_da, da_filt = filter_outliers(da, _t)
        t_dr, dr_filt = filter_outliers(dr, _t)
        axs[0, 1].plot(t_de, de_filt, label=labels[2], color=action_colors[i],
                       linestyle=ref_line_styles[i], linewidth=2)
        axs[0, 1].set_ylabel(r'$\delta_e \:\: [{deg}]$', fontsize=fontsize)
        axs[1, 1].plot(t_da, da_filt, color=action_colors[i],
                       linestyle=ref_line_styles[i], linewidth=2)
        axs[1, 1].set_ylabel(r'$\delta_a \:\: [{deg}]$', fontsize=fontsize)
        axs[2, 1].plot(t_dr, dr_filt, color=action_colors[i],
                       linestyle=ref_line_styles[i], linewidth=2)
        axs[2, 1].set_ylabel(r'$\delta_r \:\: [{deg}]$', fontsize=fontsize)

    axs[2, 0].set_xlabel('Time [s]', fontsize=fontsize)
    axs[2, 1].set_xlabel('Time [s]', fontsize=fontsize)
    leg_lines = []
    leg_labels = []
    for ax in fig.axes:
        axLine, axLegend = ax.get_legend_handles_labels()
        leg_lines.extend(axLine)
        leg_labels.extend(axLegend)

    fig.legend(leg_lines, leg_labels, loc='upper center',
               ncol=5, mode='expand', bbox_to_anchor=(0.11, 0.68, 0.8, 0.25), fontsize=13)
    fig.suptitle(faultname, fontsize=fontsize)

    # plt.show()
    if figname is not None:
        # figures_filtered_actions; figures_different_input, #figures_stability, figures; figures_trajectory_comparison
        fig_path = Path(os.getcwd()) / Path('figures_withoutSuptitle')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        figName = fig_path / Path(figname + '.pdf')
        # fig.write_image(str(figname))
        fig.savefig(figName, dpi=300, bbox_inches='tight', format='pdf')

    # record statistics
    # faultname = f"{faults[f]}[AvgnMAE {np.round(np.mean(all_nmae), 2)}% |AvgSm {np.round(np.mean(all_sm), 2)} Rad.Hz]"
    # stats = Stats(
    #     nmae=np.round(np.mean(all_nmae), 2),
    #     nmae_std=np.round(np.std(all_nmae), 2),
    #     sm=np.round(np.mean(all_sm), 2),
    #     sm_std=np.round(np.std(all_sm), 2),
    # )

    # toml_path = Path(os.getcwd()) / Path('ppo_statistics')
    # toml_name = toml_path / Path('stats_traj_comparison.toml')
    # if not os.path.exists(toml_path):
    #     os.makedirs(toml_path)

    # stats_dict = {faults[f]: stats.__dict__}
    # with open(toml_name, 'a+', encoding='utf-8') as f:
    #     f.write('\n\n')
    #     toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

    # f.close()
