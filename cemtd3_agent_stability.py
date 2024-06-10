
from plotters.plot_utils import plot
from parameters_es import ESParameters
from evaluation_es_utils import find_logs_path, gen_eval_refs, load_agent, load_cov
from environments.config import select_env
from core_algorithms.utils import calc_nMAE, calc_smoothness, load_config
from core_algorithms.td3 import Actor
from core_algorithms.replay_memory import IdentificationBuffer, ReplayMemory
from core_algorithms.model_identification import RLS
from tqdm import tqdm
import numpy as np
import argparse
import torch.multiprocessing as mp
import copy
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.font_manager as font_manager

fontpath = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf'
prop = font_manager.FontProperties(fname=fontpath)
mplt.rcParams['font.family'] = prop.get_name()


parser = argparse.ArgumentParser()

# ***** Arguments *****#
parser.add_argument('--env-name', help='Environment to be used: (PHLab)',
                    type=str, default='PHlab_attitude_nominal')
parser.add_argument(
    '--agent-name', help='Path to the agent to be evaluated', type=str, default='CEM_TD3Buffers_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h64_new_covSaving_pop50_parents10_seed42')
parser.add_argument('--seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument(
    '--use-mu', help='Use mu agent from CEM-TD3', action='store_true')
parser.add_argument('--use-best-mu', action='store_true', help='Use best mu agent from CEM-TD3')
parser.add_argument('--use-best-elite', action='store_true', help='Use best elite agent from CEM-TD3')
parser.add_argument('--disable-cuda', help='Disables CUDA',
                    action='store_true', default=False)
parser.add_argument('--num-trials', type=int, default=2)
parser.add_argument('--generate-sol', help='Generate solution',
                    action='store_true', default=False)
parser.add_argument('--save-plots', help='Save plots', action='store_true')
parser.add_argument('--mem-size', type=int, default=100000)
# ********************* #

parsed_args = parser.parse_args()
env = select_env(parsed_args.env_name)
_, _, fault_name = parsed_args.env_name.split('_')
t_max = 80
env.set_eval_mode(t_max=t_max)
params = ESParameters(parsed_args, init=True)


def find_online_logs_path(logs_name: str = './online_eval_logs'):
    cwd = os.getcwd()
    if not cwd.endswith('ftc'):
        pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = pwd

    online_logs = cwd / Path(logs_name)
    logs_name = logs_name.lower()
    if online_logs.is_dir():
        return online_logs
    return None


def fig_save_path(dir):
    logs_dir = find_logs_path(dir)
    path_ = logs_dir / Path('stability')
    return path_


def load_actor(dir, type):
    logs_dir = find_logs_path(dir)
    model_config = load_config(logs_dir)
    params.update_from_dict(model_config)
    setattr(parsed_args, type, True)
    agent = load_agent(logs_dir, params, parsed_args)
    print(">> Agent loaded successfully!")
    return agent


class LinearModel(object):
    """ Defines a linear model with matrices A and B: y=Ax+B. """

    def __init__(self, state_size, input_size, learning_rate):
        self.lr = learning_rate
        self.h = state_size
        self.inp = input_size
        self.l2_penalty = 2.0
        # A: h x h:
        # B: h x inp:
        np.random.seed(seed=10)
        A = np.random.randn(state_size, state_size)

        Q, _ = np.linalg.qr(A)
        D = np.diag(-1 + 0.5*np.random.randn(state_size,))
        self.A = Q@D@Q.T
        self.B = np.random.randn(input_size, state_size)

    def predict(self, state: np.ndarray, input: np.ndarray):
        # state: N x h:
        # input: N x inp:
        # x' = Ax + Bu
        return state.dot(self.A) + input.dot(self.B)

    def mean_squared_error(self, y, y_hat):
        v = y - y_hat
        return np.mean(np.linalg.norm(v, axis=1)**2)/2, v

    def update(self, state, input, next_state):
        # update matrix A via gradient descent:
        pred_next_state = self.predict(state, input)
        loss, err = self.mean_squared_error(next_state, pred_next_state)
        dA = (-(state.T).dot(err) + self.l2_penalty * np.sign(self.A))/N
        dB = (-(input.T).dot(err) + self.l2_penalty * np.sign(self.B))/N
        self.A -= self.lr * dA
        self.B -= self.lr * dB
        return loss


if __name__ == "__main__":

    # load an agent

    # set up an environment

    # set up a replay buffer:

    # run the agent in the environment to collect data and experiences:

    # Use the experiences to update a model of the environment. x' = f(x, a) = Ax + Bu for linear model:

    # 33
    # 1. to study the stability of an agent:
    #     1.1. combine the actor and the environment into a single  feedforward linear model
    #     1.2. with input the reference state at each timestep and output the controlled states:
    #    X' = AX:
    #     1.3. train the model with a linear regression update method:
    #     1.4. Look for the eigen value for stability check:
    # np.random.seed(42)
    state_size = 3  # theta, phi, beta:
    input_size = 3
    lr = 0.05
    N = 64  # batch size
    # input_size = 3 == output size:
    # A = np.random.randn(state_size, state_size)
    # A = np.random.randn(state_size, state_size)
    # B = np.random.randn(state_size, input_size)
    model = LinearModel(state_size, input_size, lr)
    # C = np.random.randn(input_size, state_size)

    # y = Cx
    print(model.A)
    v = np.linalg.eigvals(model.A)
    print(">> Eigen-values of A: ", v)

    buffer = IdentificationBuffer(parsed_args.mem_size)
    buffer.reset()
    if parsed_args.use_best_mu:
        agent_type = 'use_best_mu'
    elif parsed_args.use_mu:
        agent_type = 'use_mu'
    elif parsed_args.use_best_elite:
        agent_type = 'use_best_elite'
    else:
        agent_type = 'use_elite'
    agent = load_actor(parsed_args.agent_name, type=agent_type)
    # print(agent.parameters())

    max_num_episodes = 5  # 5
    losses = []
    for episode in range(max_num_episodes):
        print(">> Episode: ", episode+1)
        done = False
        obs, _ = env.reset()
        episode_loss = []
        while not done:
            # current_state: 1 x 3
            current_state = env.get_controlled_state()

            refs = np.deg2rad(np.array([ref(env.t)
                                        for ref in env.ref]).flatten())
            action = agent.select_action(obs)
            action = np.clip(action, -1, 1)

            next_obs, reward, done, _ = env.step(action.flatten())
            # next state:
            next_state = env.get_controlled_state()
            # add to the buffer:
            transition = (refs, current_state, next_state)
            buffer.push(*transition)
            obs = next_obs
            if done:
                env.reset()

        env.close()
        if buffer.__len__() > 1000:
            # train the model:

            for _ in tqdm(range(int(N/16)), desc='Regression', colour='blue'):
                batch = buffer.sample(N)
                u = batch[0].cpu().numpy()
                current_x = batch[1].cpu().numpy()
                next_x = batch[2].cpu().numpy()

                l = model.update(state=current_x, input=u,
                                 next_state=next_x)
                episode_loss.append(l)

        losses.append(np.mean(np.asarray(episode_loss)))

    print(losses)

    # fig_path = fig_save_path(parsed_args.agent_name) / \
    #     Path(agent_type) / Path(fault_name)

    fig_path = Path(os.getcwd()) / Path('figures_withoutSuptitle')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # 2. to study the stability of the environment:
    # get eigen values of A

    print(model.A)
    # print(np.sign(model.A))
    v = np.linalg.eigvals(model.A)
    print(">> Eigen-values of A: ", v)
    # plot eigen values:
    fig_eign, ax = plt.subplots(1, 1)

    ax.scatter(np.real(v), np.imag(v), marker='x',
               color='r', label='Eigen values', s=500)
    ax.set_xlabel('Real', fontsize=13)
    ax.set_ylabel('Imaginary', fontsize=13)
    ax.grid()
    fig_eign.suptitle('Eigen values of System Matrix A', fontsize=14)

    name_eig = f'{parsed_args.env_name}_cemtd3_eigen_values'
    figname = fig_path / Path(name_eig+'.pdf')
    fig_eign.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
    # fig_name = fig_path / Path('eigen_values.png')
    # fig_eign.savefig(fname=fig_name, dpi=300, format='png')

    # print(losses[0])

    # plt.plot(losses)
    # plt.show()
    # controllability matrix:
    C_m = [model.B.T, model.A@model.B.T, model.A@model.A@model.B.T]
    C_m = np.concatenate(C_m, axis=1)
    singular_val = np.linalg.svd(C_m)[1]
    print(">> Controllability: ", singular_val)

    # simulate step response:
    times = 100
    input_sequence = np.zeros((times, input_size))
    input_sequence[:, 0] = 0.0
    print(input_sequence[0])

    X_d = np.zeros((times, state_size))
    Y_d = np.zeros((times, input_size))
    # output_response = np.zeros((times, state_size))
    initial_obs, _ = env.reset()
    initial_state = env.get_controlled_state()
    print(initial_state)
    for i in range(times):
        if i == 0:
            X_d[i] = initial_state.flatten()
            Y_d[i] = initial_state.flatten()
            x = model.predict(initial_state.reshape(1, -1),
                              input_sequence[i].reshape(1, -1))
            # x = initial_state @ A + input_sequence[i].reshape(1, -1) @ B.T
        else:
            X_d[i] = x.flatten()
            Y_d[i] = x.flatten()
            x = model.predict(x, input_sequence[i].reshape(1, -1))
            # x = x @ A + input_sequence[i].reshape(1, -1) @ B.T
    X_d[-1] = x.flatten()
    Y_d[-1] = x.flatten()

    # #  output_response[i] = x.flatten()

    # # # output_response[-1] = x.flatten()

    # # # print(output_response[:, 3])
    fig_response, ax2 = plt.subplots(3, 1)
    ax2[0].plot(X_d[:, 0], label=r'$\theta$', linewidth=2, color='g')
    ax2[1].plot(X_d[:, 1], label=r'$\phi$', linewidth=2, color='g')
    ax2[2].plot(X_d[:, 2], label=r'$\beta$', linewidth=2, color='g')
    ax2[2].set_xlabel('Time steps', fontsize=13)
    ax2[0].set_ylabel(r'$\theta (rad)$', fontsize=13)
    ax2[1].set_ylabel(r'$\phi (rad)$', fontsize=13)
    ax2[2].set_ylabel(r'$\beta (rad)$', fontsize=13)
    ax2[0].grid()
    ax2[1].grid()
    ax2[2].grid()

    fig_response.suptitle('Attitude Response', fontsize=14)
    name_response = f'{parsed_args.env_name}_cemtd3_attitude_response'
    figname = fig_path / Path(name_response+'.pdf')
    fig_response.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
