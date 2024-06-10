import copy
import os
from pathlib import Path
from matplotlib import pyplot as plt
import toml
from tqdm import tqdm
from core_algorithms.utils import calc_nMAE, calc_smoothness
from plotters.plot_utils import plot, plot_attitude_response
from ppo_continuous_actions import Agent, make_env
from environments.config import select_env
import torch
import gymnasium as gym
import argparse
from evaluation_es_utils import Stats, gen_eval_refs
import numpy as np

# *** Parse arguments ***#


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


parser = argparse.ArgumentParser(description='PPO')
parser.add_argument(
    '--gym-id', default='PHlab_attitude_nominal', type=str, help='environment name')
parser.add_argument('--use-second-env', action='store_true', default=False, help='wheter to use a second env for evaluation.')
parser.add_argument(
    '--gym2-id', default='PHlab_attitude_nominal', type=str, help='second environment name.')
parser.add_argument('--switch-time', type=int, default=10, help='time at which to switch to the second environment (fault occurance timestep)')
parser.add_argument('--t-max', type=int, default=80, help='maximum time for evaluation')
parser.add_argument('--add-sm-to-reward',
                    action='store_true', default=False, help='adding smoothness metric to the reward function')
parser.add_argument('--use-scaled-obs', action='store_true', default=False, help='scale the observed states.')
parser.add_argument('--seed', default=7, type=int, help='seed')
parser.add_argument('--num-envs', default=1, type=int, help='number of envs')
parser.add_argument('--save-results', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num-trials', default=2, type=int, help='number of evaluation')
parser.add_argument('--fdd', type=str, default='none')
# [0, 12, 3, -4, -8, 2], [0, 20, 10, 1, 0, -2]
parser.add_argument(
    '--amp-theta', default=[0, 12, 3, -4, -8, 2], type=list_of_ints, help='reference pitch angles')

# [2, -2, 2, 10, 2, -6], [0, 0, 0, 40, 2, -6]
parser.add_argument(
    '--amp-phi', default=[2, -2, 2, 10, 2, -6], type=list_of_ints, help='reference roll angles')
parser.add_argument('--without-beta', action='store_true', default=False, help='whether the side-slip angle is associated with the observed states.')
parser.add_argument('--filter-action', action='store_true', default=False, help='filter the actions for plotting.')
parser.add_argument('--agent-name', default='PHlab_attitude_nominal__ppo_continous_actions__7__1707401136_addsmtorw_maxdiffsm_False__multipleEnvs_True__totalTimesteps_10000000.pkl', required=True, help='Agent name to evaluate')

# parser.add_argument(
#     '--max-theta', default=max(np.abs([0, 12, 3, -4, -8, 2])), type=float)
# parser.add_argument(
#     '--max-phi', default=max(np.abs([2, -2, 2, 10, 2, -6])), type=float)

#**** Trained agents on fault scenarios.
faults_agents = {
    'ice': 'PHlab_attitude_ice__ppo_continous_actions__1__1708075136_addsmtorw_maxdiffsm_True__multipleEnvs_False_singleENV__totalTimesteps_2000000.pkl',
    'cg-shift': 'PHlab_attitude_cg-shift__ppo_continous_actions__1__1708075188_addsmtorw_maxdiffsm_True__multipleEnvs_False_singleENV__totalTimesteps_2000000.pkl',
    'be': 'PHlab_attitude_be__ppo_continous_actions__1__1708075274_addsmtorw_maxdiffsm_True__multipleEnvs_False_singleENV__totalTimesteps_2000000.pkl',
    'be2': 'PHlab_attitude_be__ppo_continous_actions__7__1710231749_addsmtorw_maxdiffsm_False__multipleEnvs_False_singleENV__totalTimesteps_2000000__useScaledObs_False.pkl',
    'jr': 'PHlab_attitude_jr__ppo_continous_actions__1__1708075329_addsmtorw_maxdiffsm_True__multipleEnvs_False_singleENV__totalTimesteps_2000000.pkl',
    'sa': 'PHlab_attitude_sa__ppo_continous_actions__1__1708075397_addsmtorw_maxdiffsm_True__multipleEnvs_False_singleENV__totalTimesteps_2000000.pkl'
}

if __name__ == "__main__":
    args = parser.parse_args()
    args.max_theta = max(np.abs(args.amp_theta))
    args.max_phi = max(np.abs(args.amp_phi))
    # TODO: top 1-2
    # run_name = 'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl'
    # agent trained on multiple environments:
    #*** Agent name to evaluate
    run_name = args.agent_name
    # 'PHlab_attitude_nominal__ppo_continous_actions__7__1707401136_addsmtorw_maxdiffsm_False__multipleEnvs_True__totalTimesteps_10000000.pkl'
    if args.without_beta:
        run_name = 'PHlab_attitude_nominal__ppo_continous_actions__1__1711972191_addsmtorw_maxdiffsm_False__multipleEnvs_False_singleENV__totalTimesteps_2000000__useScaledObs_False__NoBeta.pkl'
    # run_name = 'PHlab_attitude_nominal__ppo_continous_actions__4__1706947059.pkl'
    # run_name = 'PHlab_attitude_nominal__ppo_singleEnv__7__1703239182.pkl'
    # run_name = 'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction_add_sm_to_rewarf_func__7__1703496554.pkl'
    # run_name = 'PHlab_attitude_nominal__ppo_continous_actions__1__1707370977_addsmtorw_fftsm_True.pkl'
    # run_name = 'PHlab_attitude_nominal__ppo_continous_actions__1__1707386696_addsmtorw_maxdiffsm_True.pkl'
    # TODO: top 1
    # run_name = 'PHlab_attitude_nominal__ppo_continous_actions__7__1707401136_addsmtorw_maxdiffsm_False__multipleEnvs_True__totalTimesteps_10000000.pkl'
    # run_name = "PHlab_attitude_nominal__ppo_continous_actions__7__1707459849_addsmtorw_maxdiffsm_False__multipleEnvs_False__totalTimesteps_10000000.pkl" # nominal env (vectorized envs for training)
    # run_name = "PHlab_attitude_nominal__ppo_continous_actions__7__1707554173_addsmtorw_maxdiffsm_False__multipleEnvs_False_singleENV__totalTimesteps_10000000.pkl"
    # TODO: with scaled-obs
    # run_name = "PHlab_attitude_nominal__ppo_continous_actions__1__1708691837_addsmtorw_maxdiffsm_False__multipleEnvs_False_singleENV__totalTimesteps_2000000__useScaledObs_True.pkl"
    # run_name = "PHlab_attitude_nominal__ppo_continous_actions__7__1708706240_addsmtorw_maxdiffsm_False__multipleEnvs_False_singleENV__totalTimesteps_2000000__useScaledObs_True.pkl"
    if args.fdd != 'none':
        run_name = faults_agents[args.fdd]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # ****** Define Env ******#
    env = select_env(
        environment_name=args.gym_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward,
        use_scaled_obs=args.use_scaled_obs,
        without_beta=args.without_beta
    )
    env.set_eval_mode(t_max=args.t_max)
    t_max = args.t_max
    if args.use_second_env:
        env2 = select_env(
            environment_name=args.gym2_id,
            conform_with_sb=True,
            add_sm_to_reward=args.add_sm_to_reward,
            use_scaled_obs=args.use_scaled_obs,
            without_beta=args.without_beta
        )
        env2.set_eval_mode(t_max=args.t_max)
        total_timesteps = int(args.t_max//0.01)
        switch_timestep = int(args.switch_time//0.01)
        remaining_timesteps = int(total_timesteps-switch_timestep)

    # make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name)
    #      for i in range(args.num_envs)
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id=args.gym_id, seed=args.seed+i, idx=i, capture_video=False, run_name=run_name, eval=True, use_scaled_obs=args.use_scaled_obs, without_beta=args.without_beta) for i in range(args.num_envs)])
    # ***** Load agent ***** #
    agent = Agent(envs=envs, eval=True).to(device)
    path = 'agents/' + run_name
    agent.load_state_dict(torch.load(path))

    print("Agent loaded from: ", path)
    # ****** Generate references ******#
    # amp_theta = [0, 12, 3, -4, -8, 2]
    # max_theta = max(np.abs(amp_theta))
    # amp_phi = [2, -2, 2, 10, 2, -6]
    # max_phi = max(np.abs(amp_phi))
    user_eval_refs = gen_eval_refs(
        amp_theta=args.amp_theta,
        amp_phi=args.amp_phi,
        t_max=t_max,
        max_theta=args.max_theta,
        max_phi=args.max_phi,
        num_trails=args.num_trials
    )

    def eval_agent(**kwargs):
        x_ctrl_lst = []
        errors = []
        ref_lst = []
        t_lst = []

        state_lst, rewards, action_lst = [], [], []
        obs, _ = env.reset(**kwargs)
        if args.use_second_env:
            env2.reset(**kwargs)
            for s in range(total_timesteps):
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

                if s == switch_timestep:
                    env2.t = copy.deepcopy(env.t)
                    env2.x = copy.deepcopy(env.x)
                    env2.obs = copy.deepcopy(env.obs)
                    env2.last_u = copy.deepcopy(env.last_u)
                    break
            for s in range(remaining_timesteps):
                action = agent.get_action_and_value(
                    torch.FloatTensor(obs))[-1].detach().numpy()
                ref_value = np.deg2rad(
                    np.array([ref(env2.t) for ref in env2.ref]).flatten())
                obs, reward, done, truncated, info = env2.step(action)
                action_lst.append(env2.last_u)
                state_lst.append(env2.x)
                x_ctrl_lst.append(env2.get_controlled_state())

                ref_lst.append(ref_value)
                errors.append(ref_value - x_ctrl_lst[-1])
                rewards.append(reward)
                t_lst.append(env2.t)

        else:
            done = False
            truncated = False
            steps = 0
            while not (done or truncated):
                action = agent.get_action_and_value(
                    torch.FloatTensor(obs))[-1].detach().numpy()
                ref_value = np.deg2rad(
                    np.array([ref(env.t) for ref in env.ref]).flatten())
                obs, reward, done, truncated, info = env.step(action)
                action_lst.append(env.last_u)
                state_lst.append(env.x)
                x_ctrl_lst.append(env.get_controlled_state())

                steps += 1

                ref_lst.append(ref_value)
                errors.append(ref_value - x_ctrl_lst[-1])
                rewards.append(reward)
                t_lst.append(env.t)

        errors = np.asarray(errors)
        actions = np.array(action_lst)
        smoothness = calc_smoothness(actions, plot_spectra=False)
        fitness = np.sum(rewards) + smoothness


        # format data:
        rewards = np.asarray(rewards).reshape((-1, 1))
        ref_values = np.array(ref_lst)
        t_lst = np.asarray(t_lst).reshape((-1, 1))
        data = np.concatenate(
            (ref_values, actions, state_lst, rewards, t_lst), axis=1)

        # calculate nMAE:
        nmae = calc_nMAE(errors)
        return None, data, nmae, smoothness

    # ****** Evaluate agent ******#
    agent_nmae_lst, agent_sm_lst = [], []
    for i in tqdm(range(args.num_trials+1), total=args.num_trials):
        ref_t = user_eval_refs[i]
        user_refs = {
            'theta_ref': ref_t[0],
            'phi_ref': ref_t[1]
        }
        _, data, nmae, sm = eval_agent(user_refs=user_refs)
        agent_nmae_lst.append(nmae)
        agent_sm_lst.append(sm)
        # print(sm)

    nmae = np.mean(agent_nmae_lst)
    nmae_std = np.std(agent_nmae_lst)
    smoothness = np.median(agent_sm_lst)
    sm_std = np.std(agent_sm_lst)

    stats = Stats(nmae, nmae_std, smoothness, sm_std)
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

    if args.use_second_env:
        switch_time = args.switch_time
        _, _, f1 = args.gym_id.split('_')
        _, _, f2 = args.gym2_id.split('_')
        faultName = f"{faults[f1]}_to_{faults[f2]}_at_{switch_time}_sec"
        figname = f"{faultName}_{run_name.split('.')[0]}"
    else:
        _, _, fault_name = args.gym_id.split('_')
        faultName = faults[fault_name]
        agent_name, _ = run_name.split('.')
        figname = f"{faultName}_{agent_name}"
        switch_time = None

    fig, _ = plot(
        data, name=f'nMAE: {stats.nmae:0.2f}% - Smoothness: {stats.sm:0.2f} rad.Hz', fault=faultName, fig_name=figname, switch_time=switch_time, filter_action=args.filter_action
    )

    err = np.round(stats.nmae, 2)
    sm = np.round(stats.sm, 2)
    plot_attitude_response(
        data, name=run_name, fault=fault_name, nmae=err, sm=sm)
    plt.show()

    # toml_path = Path(os.getcwd()) / Path('ppo_statistics')
    # toml_name = toml_path / Path('stats_withoutBeta.toml')
    # if not os.path.exists(toml_path):
    #     os.makedirs(toml_path)

    # stats_dict = {faultName: stats.__dict__}
    # with open(toml_name, 'a+', encoding='utf-8') as f:
    #     f.write('\n\n')
    #     toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

    # f.close()
