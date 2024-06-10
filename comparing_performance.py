import copy
import os
from pathlib import Path
# from matplotlib import pyplot as plt
# import toml
from tqdm import tqdm
from core_algorithms.utils import Episode, calc_nMAE, calc_smoothness
from evaluation_es_utils import gen_eval_refs
from parameters_es import ESParameters
from plotters.plot_utils import plot_comparative_analysis
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
parser.add_argument('--use-second-env', action='store_true', default=False)
parser.add_argument(
    '--gym2-id', default='PHlab_attitude_be', type=str, help='environment')
parser.add_argument('--switch-time', type=int, default=10)
parser.add_argument('--t-max', type=int, default=80)
parser.add_argument('--add-sm-to-reward',
                    action='store_true', default=False)
parser.add_argument('--use-scaled-obs', action='store_true', default=False)
parser.add_argument('--seed', default=7, type=int, help='seed')
parser.add_argument('--num-envs', default=1, type=int, help='number of envs')
parser.add_argument('--save-results', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num-trials', default=2, type=int)
# parser.add_argument('--fdd', type=str, default='none')
# [0, 12, 3, -4, -8, 2], [0, 20, 10, 1, 0, -2]
parser.add_argument(
    '--amp-theta', default=[0, 12, 3, -4, -8, 2], type=list_of_ints)

# [2, -2, 2, 10, 2, -6], [0, 0, 0, 40, 2, -6]
parser.add_argument(
    '--amp-phi', default=[2, -2, 2, 10, 2, -6], type=list_of_ints)
parser.add_argument('--agent-name', type=str, default='PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl', help='Agent name', required=True)
# parser.add_argument()


params = ESParameters(init=True)




if __name__ == "__main__":
    args = parser.parse_args()

    args.max_theta = max(np.abs(args.amp_theta))
    args.max_phi = max(np.abs(args.amp_phi))
    # TODO: top 1-2

    run_name = args.agent_name
    'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl'

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # ****** Define Env ******#
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id=args.gym_id, seed=args.seed+i, idx=i, capture_video=False, run_name=run_name, eval=True, use_scaled_obs=args.use_scaled_obs) for i in range(args.num_envs)])

    # ***** Load agent ***** #
    agent = Agent(envs=envs, eval=True).to(device)
    path = 'agents/' + run_name
    agent.load_state_dict(torch.load(path))

    print("Agent loaded from: ", path)
    # ****** Generate references ******#
    user_eval_refs = gen_eval_refs(
        amp_theta=args.amp_theta,
        amp_phi=args.amp_phi,
        t_max=args.t_max,
        max_theta=args.max_theta,
        max_phi=args.max_phi,
        num_trails=args.num_trials
    )

    env1 = select_env(
        environment_name=args.gym_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward,
        use_scaled_obs=args.use_scaled_obs,
    )
    env1.set_eval_mode(t_max=args.t_max)
    t_max = args.t_max
    # if args.use_second_env:
    env2 = select_env(
        environment_name=args.gym2_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward,
        use_scaled_obs=args.use_scaled_obs,
    )
    env2.set_eval_mode(t_max=args.t_max)

    envs = [env1, env2]

    def eval_agent(**kwargs):
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
            mult_nmae.append(nmae)
            actions = np.array(action_lst)
            sm = calc_smoothness(actions, plot_spectra=False)
            mult_sm.append(sm)
            ref_values = np.array(ref_lst)
            t_lst = np.asarray(t_lst).reshape((-1, 1))
            data = np.concatenate(
                (ref_values, actions, state_lst, t_lst), axis=1)
            mult_data.append(data)

        # mult_data = np.concatenate(mult_data, axis=0)
        mult_data = np.asarray(mult_data)

        return mult_data, mult_nmae, mult_sm

    agent_nmae_lst, agent_sm_lst = [], []
    for i in tqdm(range(args.num_trials+1), total=args.num_trials+1):
        ref_t = user_eval_refs[i]
        user_refs = {
            'theta_ref': ref_t[0],
            'phi_ref': ref_t[1]
        }

        data, nmae, sm = eval_agent(user_refs=user_refs)
        agent_nmae_lst.append(nmae)
        agent_sm_lst.append(sm)
        # print(sm)
    agent_nmae_lst = np.asarray(agent_nmae_lst)
    agent_sm_lst = np.asarray(agent_sm_lst)
    nmae = np.round(np.mean(agent_nmae_lst, axis=0), 2)
    nmae_std = np.std(agent_nmae_lst, axis=0)
    smoothness = np.round(np.median(agent_sm_lst, axis=0), 2)
    sm_std = np.std(agent_sm_lst, axis=0)

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
    _, _, f2 = args.gym2_id.split('_')

    faultname = f"[nMAE {nmae[0]}%|Sm {smoothness[0]} Rad.Hz] vs [nMAE {nmae[1]}%|Sm {smoothness[1]} Rad.Hz]"
    name = f"{faults[f1]}_vs_{faults[f2]}"
    figname = f"{name}_{run_name.split('.')[0]}"

    # print(data.shape, data)
    print(agent_nmae_lst)
    print(agent_sm_lst)
    print(nmae, nmae_std, smoothness, sm_std)

    fig = plot_comparative_analysis(data, faultname, figname)
