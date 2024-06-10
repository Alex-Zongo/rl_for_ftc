
from core_algorithms.replay_memory import IdentificationBuffer
from ppo_continuous_actions import Agent as NormalController
# from plotters.plot_utils import plot
from fdi_adaptation2 import SmoothFilter
from core_algorithms.fdi import FDI
from environments.config import select_env
import torch
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import argparse
from distutils.util import strtobool
import os
from pathlib import Path
import gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib as mplt
import matplotlib.font_manager as font_manager

fontpath = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf'
prop = font_manager.FontProperties(fname=fontpath)
mplt.rcParams['font.family'] = prop.get_name()

controllers_names = {
    0: 'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl',
    1: 'PHlab_attitude_nominal__ppo_continous_actions__7__1707401136_addsmtorw_maxdiffsm_False__multipleEnvs_True__totalTimesteps_10000000.pkl',
}

fdi_continuously_trained = {
    0: "fdi_continuous_training__1__1707819572__totalTimesteps_2000000.pkl",
}

filter_names = {
    # TODO: using smfft :
    0: "fdi_adaptation2__1__1708140439__totalTimesteps_2000000_Epochs20_obsAndfdi_SampledKp_smfftAndRew0.8__fdiMinusCaction_SingleNormENV.pkl",
    1: "fdi_adaptation2__1__1708329330__totalTimesteps_2000000_Epochs20_obsAndfdi_SampledKp_smfftAndRew0.95__fdiMinusCaction_SingleNormENV.pkl"
}


class LinearModel(object):
    def __init__(self, state_size, input_size, learning_rate):
        self.lr = learning_rate
        self.h = state_size
        self.inp = input_size
        self.l2_penalty = 2.0
        # A: h x h:
        # B: h x inp:
        np.random.seed(seed=42)
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


def make_env(gym_id, seed, add_sm=False, eval=False, t_max=20):
    def thunk():
        if gym_id.startswith('PHlab'):
            env = select_env(
                environment_name=gym_id,
                conform_with_sb=True,
                add_sm_to_reward=add_sm
            )
            env.t_max = t_max  # setting t_max to 10s
            if eval:
                env.set_eval_mode(t_max=80)
        elif gym_id.startswith('LunarLander'):
            env = gym.make(
                gym_id,
                continuous=True,
                gravity=-9.8,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5,
            )
            # print(env.action_space)
        elif gym_id.startswith('Pendulum'):
            env = gym.make(
                gym_id,
                g=9.81,
            )
        elif gym_id.startswith('BipedalWalker'):
            env = gym.make(
                gym_id,
                # g=9.81,
            )
        else:
            env = gym.make(gym_id)

        # keep track of the episode cumulative reward and episode length (info['episode']['r'] and info['episode']['l'])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = Monitor(env)
        # env = ActionLoggingWrapper(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        check_env(env)
        return env

    return thunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gym-id', type=str,
                        default='PHlab_attitude_nominal', help='environment')
    parser.add_argument('--gym-id2', type=str,
                        default='PHlab_attitude_cg-shift', help='environment2')
    parser.add_argument('--total-timesteps', type=int, default=104000)
    parser.add_argument('--num-steps', type=int, default=8000)
    parser.add_argument('--num-trials', type=int, default=2)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument(
        '--eval', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--add-sm-to-reward',
                        action='store_true', default=False)
    parser.add_argument('--capture-video',
                        type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--t-max', type=int, default=80)
    parser.add_argument('--mem-size', type=int, default=2000000)
    args = parser.parse_args()
    args.num_updates = int(args.total_timesteps // args.num_steps)
    return args


if __name__ == "__main__":
    args = parse_args()
    args.use_relu = False
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # set up an environment
    env = select_env(
        environment_name=args.gym_id,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward
    )
    env.t_max = args.t_max  # setting t_max to 10s
    env2 = select_env(
        environment_name=args.gym_id2,
        conform_with_sb=True,
        add_sm_to_reward=args.add_sm_to_reward
    )
    # env.set_eval_mode()
    # env2.set_eval_mode(t_max=args.t_max)
    # args.eval = True
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id=args.gym_id, seed=args.seed+i, add_sm=args.add_sm_to_reward, eval=args.eval, t_max=args.t_max) for i in range(args.num_envs)])

    # load an agent
    # ****** Load FDI ******#
    fdi = FDI(env=envs, args=args).to(device)
    fdi_path = 'agents/' + fdi_continuously_trained[0]
    fdi.load_state_dict(torch.load(fdi_path))
    print("FDI loaded from: ", fdi_path)

    # ****** Load Controller ******#
    controller = NormalController(envs=envs, eval=args.eval).to(device)
    controller_path = 'agents/' + controllers_names[0]
    controller.load_state_dict(torch.load(controller_path))
    print("Controller loaded from: ", controller_path)

    # ****** Load Filter ******#

    sm_filter = SmoothFilter(envs=envs, args=args).to(device)
    sm_filter_path = 'agents/' + filter_names[0]
    sm_filter.load_state_dict(torch.load(sm_filter_path))
    print("Filter loaded from: ", sm_filter_path)

    # ***** Load Linearized Model ***** #
    state_size = 3  # theta, phi, beta:
    input_size = 3
    lr = 5e-2  # 0.05
    N = 64  # batch size

    model = LinearModel(state_size, input_size, lr)
    print(model.A)
    v = np.linalg.eigvals(model.A)
    print(">> Eigen-values of A: ", v)

    # fig path:
    fig_path = Path(os.getcwd()) / Path('figures_withoutSuptitle')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # set up a replay buffer:
    buffer = IdentificationBuffer(args.mem_size)
    buffer.reset()
    # run the agent in the environment to collect data and experiences:
    losses = []
    for up in range(args.num_updates):
        ep_losses = []
        obs, _ = env.reset()
        for i in range(args.num_steps):

            curr_state = env.get_controlled_state()
            refs = np.deg2rad(np.array([ref(env.t)
                                        for ref in env.ref]).flatten())
            c_action = controller.get_action_and_value(
                torch.FloatTensor(obs))[-1]
            f, f_action = fdi.forward(
                torch.FloatTensor(obs), c_action)
            kp = sm_filter.get_sm_param_and_value(
                torch.FloatTensor(obs), f_action)[-2]
            action = c_action + kp * (f_action-c_action)

            obs, reward, done, trunc, info = env.step(
                action.detach().cpu().numpy())
            next_state = env.get_controlled_state()

            transition = (refs, curr_state, next_state)
            buffer.push(*transition)

        # Use the experiences to update a model of the environment. x' = f(x, a) = Ax + Bu for linear model:
        if buffer.__len__() > 1000:
            for _ in range(int(N/16)):
                batch = buffer.sample(N)
                u = batch[0].cpu().numpy()
                current_x = batch[1].cpu().numpy()
                next_x = batch[2].cpu().numpy()

                l = model.update(state=current_x, input=u,
                                 next_state=next_x)
                ep_losses.append(l)

        losses.append(np.mean(ep_losses))

    # 33
    # 1. to study the stability of an agent:
    #     1.1. combine the actor and the environment into a single  feedforward linear model
    #     1.2. with input the reference state at each timestep and output the controlled states:
    #    X' = AX:
    #     1.3. train the model with a linear regression update method:
    #     1.4. Look for the eigen value for stability check:
    # np.random.seed(42)

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
    name_eig = f'{args.gym_id}_ppo_eigen_values'
    figname = fig_path / Path(name_eig+'.pdf')
    fig_eign.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
    # fig_eign.show()
    # plt.show()

    C_m = [model.B.T, model.A@model.B.T, model.A@model.A@model.B.T]
    C_m = np.concatenate(C_m, axis=1)
    singular_val = np.linalg.svd(C_m)[1]
    print(">> Controllability: ", singular_val)

    # simulate a step response
    times = 100
    input_sequence = np.zeros((times, input_size))
    input_sequence[:, 0] = 0.0
    X_d = np.zeros((times, state_size))
    Y_d = np.zeros((times, input_size))

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
    # fig_response.savefig(fname=fig_path / Path('attitude_response.png'),
    #                      dpi=300, format='png')
    name_response = f'{args.gym_id}_ppo_attitude_response'
    figname = fig_path / Path(name_response+'.pdf')
    fig_response.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
    # fig_response.show()
    # plt.show()
