
from core_algorithms.model_identification import ESModelIdentification
from dataclasses import dataclass
# import toml
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, CompoundKernel, ConstantKernel, RationalQuadratic
import multiprocessing as mp
import torch
# import matplotlib.pyplot as plt
from pathlib import Path
import os
import copy
import torch.multiprocessing as mp
import argparse

import numpy as np
from tqdm import tqdm
from core_algorithms.model_identification import RLS, GaussianProcess
# from core_algorithms.replay_memory import IdentificationBuffer, ReplayMemory
from core_algorithms.td3 import Actor
from core_algorithms.utils import calc_nMAE, calc_smoothness, load_config

from environments.config import select_env
from evaluation_es_utils import find_logs_path, gen_eval_refs, load_agent_online, load_cov
from parameters_es import ESParameters
from plotters.plot_utils import plot
import ray

parser = argparse.ArgumentParser()

# ***** Arguments *****#
parser.add_argument('--env-name', help='Environment to be used: (PHLab)',
                    type=str, default='PHlab_attitude_nominal')
parser.add_argument('--use-second-env', action='store_true')
parser.add_argument('--env2-name', type=str, default='PHlab_attitude_nominal')
parser.add_argument('--switch-time', type=int, default=10)

parser.add_argument(
    '--agent_name', help='Path to the agent to be evaluated', type=str)
parser.add_argument('--seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument(
    '--use-mu', help='Use mu instead of best agent', action='store_true')
parser.add_argument('--use-best-mu', action='store_true')
parser.add_argument('--use-best-elite', action='store_true')
parser.add_argument('--disable-cuda', help='Disables CUDA',
                    action='store_true', default=False)
parser.add_argument('--num-trials', type=int, default=2)
parser.add_argument('--generate-sol', help='Generate solution',
                    action='store_true', default=False)
parser.add_argument('--save-plots', help='Save plots', action='store_true')
parser.add_argument('--save-stats', help='Save stats', action='store_true')
parser.add_argument('--t-max', type=int, default=100)
parser.add_argument('--save-trajectory',
                    help='Save trajectory', action='store_true')
parser.add_argument('--mem-size', type=int, default=100000)
parser.add_argument('--use-fdiNN', action='store_true')
parser.add_argument(
    '--amp-theta', default=[0, 12, 3, -4, -8, 2], type=list)
parser.add_argument('--amp-phi', default=[2, -2, 2, 10, 2, -6], type=list)
parser.add_argument('--filter-action', action='store_true', default=False)

# ********************* #

parsed_args = parser.parse_args()
parsed_args.max_theta = max(np.abs(parsed_args.amp_theta))
parsed_args.max_phi = max(np.abs(parsed_args.amp_phi))

env = select_env(
    environment_name=parsed_args.env_name,
    conform_with_sb=True,
)

t_max = parsed_args.t_max
env.set_eval_mode(t_max=t_max)

if parsed_args.use_second_env:
    env2 = select_env(
        environment_name=parsed_args.env2_name,
        conform_with_sb=True,
    )
    env2.set_eval_mode(t_max=parsed_args.t_max)
params = ESParameters(parsed_args, init=True)

# **** ENV setup:
params.action_dim = env.action_space.shape[0]
params.state_dim = env.observation_space.shape[0]
_, _, fault_name = parsed_args.env_name.split('_')

# load the agent or agents:
# if One agent: then load the cov matrix:
# Sample a list of agent actor for the task:


# if parsed_args.generate_sol and parsed_args.agent_name:
#     # load mu agent and covariance matrix:
#     agents = generate_agents(n_sol, params, parsed_args)
# else:
# load agents: mu, best_mu, elite and best elite
# agents_dir = [
#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_seed0",
#     # "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed42",
#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed7",

#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed0",
#     "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed0"
# ]

#**** list of agents used to evaluate the online adaptation algorithm.
n_agents_name = [
    "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed42",
    "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed0"
    "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_seed0",

    # "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_seed0","CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_seed42",
    # "CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_50iters_deep2h64_new_covSaving_seed0",
    # "CEM_PPO_SingleAgent_deep2h64_pop20_parents10_test_timesteps5000000_new_seed0",
    # "CEM_PPO_SingleAgent_deep2h64_pop20_parents10_test_timesteps5000000_new_seed42",
]

# "mu", "elite", "best_mu", "best_elite"
agents_type = ["use-best-mu", "use-best-elite", "use-mu"]
# agents_type = ["use_best_mu", "use_best_elite",
#                "use_mu", "use_elite"]


def agents_(agents_name, type):
    l = []
    for name in agents_name:
        for t in type:
            l.append((name, t))

    return l


p_comb = agents_(agents_name=n_agents_name, type=agents_type)
# p_comb = list(zip(agents_dir, agents_type))
print(p_comb)
agents = []
for dir, type in p_comb:
    logs_dir = find_logs_path(dir)
    model_config = load_config(logs_dir)
    actor_params = copy.deepcopy(params)
    actor_params.update_from_dict(model_config)
    setattr(parsed_args, type, True)
    # print(parsed_args.use)
    setattr(actor_params, 'device', torch.device("cpu"))
    actor = Actor(actor_params, init=True)
    agent = load_agent_online(logs_dir, actor_params, type)
    print(f">> Agent loaded: {dir} - {type}")

    agents.append(agent)

print(">> Number of agents: ", len(agents))
##################


# identified_model = RLS(config_dict=rls_config, env=env)
#**** Identification model: either a DNN or a Gaussian Process.
if parsed_args.use_fdiNN:
    identified_model = ESModelIdentification(env=env, config=params)
else:
    # 1.0*RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e1))
    kernel_dict = {
        "RBF":  RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
        "RationalQ": RationalQuadratic(length_scale=1.0, alpha=1.0,  length_scale_bounds=(1e-5, 1e5)),
    }

    def choose_kernel(name: str):
        if not name:
            name = "RBF"
        print("choose_kernel: ", name)
        return kernel_dict[name]
        #################
    config = dict(
        gamma=0.99,
        cov0=1e-1,
        state_size=6,
        eps_thresh=1e-3,
        seed=42,
        kernel=choose_kernel("RationalQ"),

    )
    identified_model = GaussianProcess(config_dict=config, env=env)
# load saved model if exists:
identified_model.reset()
t_horizon = 10.0  # 10
# establish the ref trajectory:
# amp1 = [0, 12, 3, -4, -8, 2]
# amp1_max = 12.0
# amp_theta = [0, -2, 4, 6, -3, 2]
# max_theta = 6.0
# amp2 = [2, -2, 2, 10, 2, -6]
# amp2_max = 10.0
# amp_phi = [2, -3, 3, -4, 5, -6]
# max_phi = 5.0
user_eval_refs = gen_eval_refs(
    amp_theta=parsed_args.amp_theta,
    amp_phi=parsed_args.amp_phi,
    max_theta=parsed_args.max_theta,
    max_phi=parsed_args.max_phi,
    t_max=t_max,
    num_trails=parsed_args.num_trials,
)
# initialize the identification model:
user_refs = {
    'theta_ref': user_eval_refs[-1][0],
    'phi_ref': user_eval_refs[-1][1],
}

obs, _ = env.reset(user_refs=user_refs)
if parsed_args.use_second_env:
    env2.reset(user_refs=user_refs)
identified_model.sync_env(env)


def parallel_predictive_control(controller):
    global agents
    global identified_model
    global t_horizon
    rewards, action_lst, times = identified_model.predictive_control(
        agents[controller], t_horizon=t_horizon)
    return sum(rewards)


@ray.remote
def parallel_predictive_control_remote(controller):
    return parallel_predictive_control(controller)


def find_online_logs_path(logs_name: str = './online_eval_logs'):
    cwd = os.getcwd()
    if not cwd.endswith('control'):
        pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = pwd

    online_logs = cwd / Path(logs_name)
    logs_name = logs_name.lower()
    if online_logs.is_dir():
        return online_logs
    return None


def save_trajectory(fault_path, data, agent_type):
    """save the actor trajectory data

    Args:
        fault_path (_type_): _description_
        data (_type_): _description_
    """
    save_path = fault_path / Path(agent_type + '_trajectory.csv')
    with open(save_path, 'w+', encoding='utf-8') as fp:
        np.savetxt(fp, data)

    print(f'Trajectory saved as: {save_path}')
    fp.close()


@dataclass
class OnlineStats:
    nmae: np.float16
    fitness: np.float16
    sm: np.float16
    n_change: np.int16


if __name__ == "__main__":
    # ctx = torch.multiprocessing.get_context('spawn')
    def soft_switch(target_agent, source_agent, tau):
        """
        Soft update of the target network parameters.
        θ_local = (1-τ)*θ_local + τ*θ_target
        """

        for target_param, param in zip(target_agent.parameters(), source_agent.parameters()):
            param.data.copy_(tau*target_param.data + (1.0-tau)*param.data)
            # target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

    def comp_reward(new_r, c_r):
        """_summary_

        Args:
            new_r (_type_): new policy reward
            c_r (_type_): current policy reward

        Returns:
            Wether to switch or not
        """
        return (new_r/c_r) <= 0.5

    ref_lst, errors, rewards, t_lst = [], [], [], []
    x_lst, x_ctrl_lst, u_lst = [], [], []
    curr_agent = copy.deepcopy(agents[0])
    step = 0
    num_of_agent_change = 0
    curr_actor_idx = 0
    if parsed_args.use_second_env:
        total_steps = int(parsed_args.t_max//0.01)

        for stp in range(total_steps):
            u_lst.append(env.last_u)
            x_lst.append(env.x)
            x_ctrl_lst.append(env.get_controlled_state())
            ref_value = np.deg2rad(
                np.array([ref(env.t) for ref in env.ref]).flatten()
            )

            action = curr_agent.select_action(obs)
            action = np.clip(action, -1, 1)
            next_obs, reward, done, _, info = env.step(action.flatten())
            if parsed_args.use_fdiNN:
                if step % 10 == 0:
                    identified_model.update(
                        state=obs,
                        action=action,
                        next_state=next_obs,
                    )
            else:
                identified_model.update(
                    state=obs[:identified_model.state_size],
                    action=action,
                    next_state=next_obs[:identified_model.state_size],
                )
            # every 5 s and after 10 s
            if step % 200 == 0 and step >= 0:
                p = mp.Pool(len(agents))
                identified_model.sync_env(env)

                r = p.map(parallel_predictive_control,
                          list(range(len(agents))))

                idx = np.argmax(r)
                if curr_actor_idx != idx and comp_reward(r[idx], r[curr_actor_idx]):
                    num_of_agent_change += 1
                    print(idx, r[idx]/r[curr_actor_idx])
                    curr_actor_idx = idx

            soft_switch(agents[curr_actor_idx],
                        curr_agent, tau=0.00005)  # 0.00005

            # save the stats:
            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env.t)

            obs = next_obs
            step += 1
            if stp == int(parsed_args.switch_time//0.01):
                env2.t = copy.deepcopy(env.t)
                env2.x = copy.deepcopy(env.x)
                env2.obs = copy.deepcopy(obs)
                env2.last_u = copy.deepcopy(env.last_u)
                # env2.u = copy.deepcopy(env.u)
                break

        for stp in range(int(total_steps-parsed_args.switch_time//0.01)):
            u_lst.append(env2.last_u)
            x_lst.append(env2.x)
            x_ctrl_lst.append(env2.get_controlled_state())
            ref_value = np.deg2rad(
                np.array([ref(env2.t) for ref in env2.ref]).flatten()
            )

            action = curr_agent.select_action(obs)
            action = np.clip(action, -1, 1)
            next_obs, reward, done, _, info = env2.step(action.flatten())
            if parsed_args.use_fdiNN:
                if step % 10 == 0:
                    identified_model.update(
                        state=obs,
                        action=action,
                        next_state=next_obs,
                    )
            else:
                identified_model.update(
                    state=obs[:identified_model.state_size],
                    action=action,
                    next_state=next_obs[:identified_model.state_size],
                )
            # every 5 s and after 10 s
            if step % 200 == 0 and step >= 0:
                p = mp.Pool(len(agents))
                identified_model.sync_env(env2)

                r = p.map(parallel_predictive_control,
                          list(range(len(agents))))
                idx = np.argmax(r)
                if curr_actor_idx != idx and comp_reward(r[idx], r[curr_actor_idx]):
                    num_of_agent_change += 1
                    print(idx, r[idx]/r[curr_actor_idx])
                    curr_actor_idx = idx

            soft_switch(agents[curr_actor_idx],
                        curr_agent, tau=0.00005)

            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env2.t)
            obs = next_obs
            step += 1
        env.close()
        env2.close()
    else:
        done = False

        while not done:
            # print(obs)
            u_lst.append(env.last_u)
            x_lst.append(env.x)
            x_ctrl_lst.append(env.get_controlled_state())
            ref_value = np.deg2rad(
                np.array([ref(env.t) for ref in env.ref]).flatten()
            )

            action = curr_agent.select_action(obs)
            action = np.clip(action, -1, 1)
            next_obs, reward, done, _, info = env.step(action.flatten())
            if parsed_args.use_fdiNN:
                if step % 10 == 0:
                    identified_model.update(
                        state=obs,
                        action=action,
                        next_state=next_obs,
                    )
            else:
                identified_model.update(
                    state=obs[:identified_model.state_size],
                    action=action,
                    next_state=next_obs[:identified_model.state_size],
                )
            # every 5 s and after 10 s
            if step % 200 == 0 and step >= 0:
                p = mp.Pool(len(agents))
                identified_model.sync_env(env)
                # print(identified_model.env.t)

                # p = ctx.Pool(4)
                r = p.map(parallel_predictive_control,
                          list(range(len(agents))))

                # r = ray.get([parallel_predictive_control_remote.remote(i) for i in range(len(agents))])
                # r = np.asarray(r)
                idx = np.argmax(r)
                if curr_actor_idx != idx and comp_reward(r[idx], r[curr_actor_idx]):
                    num_of_agent_change += 1
                    print(idx, r[idx]/r[curr_actor_idx])
                    curr_actor_idx = idx

            soft_switch(agents[curr_actor_idx],
                        curr_agent, tau=0.00005)  # 0.00005

            # save the stats:
            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)
            t_lst.append(env.t)

            obs = next_obs
            step += 1
            if done:
                env.reset()
                print(">> Env reset!")

        env.close()

    errors = np.asarray(errors)
    nmae = calc_nMAE(errors)

    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions)

    # stats:
    rewards = np.asarray(rewards).reshape((-1, 1))
    ref_values = np.array(ref_lst)
    t_lst = np.asarray(t_lst).reshape((-1, 1))
    data = np.concatenate(
        (ref_values, actions, x_lst, rewards, t_lst), axis=1)
    # stats to print:
    fitness = np.sum(rewards) + smoothness
    print(f'Episode finished after {step} steps.')

    print(f'Episode length: {t_lst[-1]} seconds.')
    print(f'Episode Agent fitness: {fitness}')
    print(f'Episode smoothness: {smoothness}')
    print(f'Episode nMAE: {nmae}')
    print(f'Number of agent change: {num_of_agent_change}')

    faults = {
        'nominal': 'Normal Flight',
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
    if parsed_args.use_second_env:

        switch_time = parsed_args.switch_time
        _, _, f1 = parsed_args.env_name.split('_')
        _, _, f2 = parsed_args.env2_name.split('_')
        faultName = f"{faults[f1]}_to_{faults[f2]}_at_{switch_time}_sec"
        figname = f"{faultName}__CEMTD3_Online_Adaptation"
    else:
        figname = f"{faults[fault_name]}__CEMTD3_Online_Adaptation"
        faultName = faults[fault_name]
        switch_time = None
    fig, _ = plot(
        data, name=f'nMAE: {nmae:0.2f}% - Smoothness: {smoothness:0.2f} rad.Hz | Num-Switch: {num_of_agent_change}', fault=fault_name, fig_name=figname, switch_time=switch_time, filter_action=parsed_args.filter_action,
    )

    fig_path = find_online_logs_path() / Path('figures')
    fault_path = fig_path / Path(fault_name)
    # if parsed_args.save_plots:
    #     if not os.path.exists(fig_path):
    #         os.mkdir(fig_path)
    #     if not os.path.exists(fault_path):
    #         os.mkdir(fault_path)
    #     fig_name = fault_path / \
    #         Path(fault_name + '.png')

    #     fig.savefig(fname=fig_name, dpi=300, format='png')
    #     plt.close()
    # else:
    #     plt.show()

    if parsed_args.save_trajectory:
        save_trajectory(fault_path, data, agent_type="online_adaptation")

    # if parsed_args.save_stats:
    #     stats = OnlineStats(
    #         nmae=nmae,
    #         fitness=fitness,
    #         sm=smoothness,
    #         n_change=num_of_agent_change,
    #     )

    #     toml_path = find_online_logs_path() / Path(fault_name)
    #     stats_dict = {fault_name: stats.__dict__}

    #     with open(os.path.join(toml_path, 'stats.toml'), 'w', encoding='utf-8') as f:
    #         f.write('\n\n')
    #         toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

    #     f.close()
