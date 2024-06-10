from functools import partial
import torch.multiprocessing as mp
import ray
from core_algorithms.cem_rl import CEM
from parameters_es import ESParameters
from core_algorithms.model_utils import activations, LayerNorm, is_lnorm_key
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, CompoundKernel, WhiteKernel
import copy
import os
import gymnasium as gym
import numpy as np
from environments.aircraftenv import AircraftEnv, printGreen, printPurple
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import torch.nn as nn


class Model(gym.Env):
    """ Online Model Identification """

    def __init__(self, config_dict: dict, env: AircraftEnv):
        super().__init__()
        self.config = config_dict
        self.env = env
        self.state_size = min(10, self.config['state_size'], env.n_obs_full)
        self.action_size = env.action_space.shape[0]

        self.dt = self.env.dt

    def reset(self):
        raise NotImplementedError

    def get_X(self, state, action):
        """ Concatenate state and action vectors. """
        state = state[:self.state_size]
        X = np.hstack([state, action])
        return X.reshape(-1, self.state_size + self.action_size)

    def update(self, state, action, next_state):
        """ Update model. (RLS) based on one state-action sample.
            Args:
                state (np.ndarray): State vector.
                action (np.ndarray): Action vector.
                next_state (np.ndarray): Next state vector."""
        raise NotImplementedError

    def predict(self, state, action):
        """ Predict the next state based on RLS Model. """
        raise NotImplementedError

    def step(self, action):
        """ Simulate a step in the local environment. """
        is_done = False

        action = self.env.scale_action(action)

        # **** CONTROLLED STATE

        # [theta, phi, beta]=[7, 6, 5]

        # x = copy.copy(self.env.x[self.ctrl_state_idx])
        # x = np.divide(x, self.scale[self.ctrl_state_idx])

        obs = copy.copy(self.env.obs[:self.state_size])
        pred_next_obs = self.predict(obs, action).flatten()
        pred_next_obs = np.multiply(
            pred_next_obs, self.scale[self.ctrl_state_idx])
        self.env.x[self.ctrl_state_idx] += pred_next_obs

        # pred_ctrl_x = self.predict(x, action).flatten()
        # pred_ctrl_x = np.multiply(pred_ctrl_x, self.scale[self.ctrl_state_idx])
        # self.env.x[self.ctrl_state_idx] = copy.copy(pred_ctrl_x)

        # Reward using clipped error
        reward = self.env.get_reward()

        # Calculate cost:
        cost = self.env.get_cost()

        # Update observation based on perfect observations & actuator state
        self.env.obs = np.hstack(
            (self.env.error, self.env.x[self.env.obs_idx]))
        self.env.last_u = action

        # Check for bounds and addd corresponding penalty for dying early
        is_done, penalty = self.env.check_bounds()
        reward += (0.2*penalty)

        # Step time
        self.env.t += self.env.dt
        self.t += self.env.dt
        self.env.last_obs = self.env.obs

        # info:
        info = {
            "ref": self.env.ref,
            "x":   self.env.x,
            "t":   self.env.t,
            "cost": cost,
        }
        return self.env.obs, reward, is_done, info

    def predictive_control(self, controller: object, t_horizon: float = 5.0, **kwargs):
        """ Predicts the trajectory of a controller (i.e., actor)
            for a finite horizon in the future.

            Args:
                controller (object): Actor object to be used during prediction.
                t_horizon (float, optional): Prediction horizon. Defaults to 5.

            Returns:
                tuple: List of rewards, list of time-steps played.
        """
        t_hor = t_horizon
        done = False
        rewards, action_lst, times = [], [], []

        # Initialise local progress:
        imagine_obs = self.env.obs
        t_start = self.env.t
        controller.eval()
        # play the future time-horizon
        while self.env.t < t_start + t_hor and not done:

            # select action:
            action = controller.select_action(imagine_obs)

            # Simulate one step in environment
            action = np.clip(action, -1, 1)

            # step
            next_obs, reward, done, _ = self.step(action.flatten())
            imagine_obs = next_obs

            # save
            rewards.append(reward)
            action_lst.append(self.env.last_u)
            times.append(self.env.t)

        rewards = np.asarray(rewards)
        action_lst = np.asarray(action_lst)
        times = np.asarray(times)

        return rewards, action_lst, times

    def sync_env(self, source_env: gym.Env, **kwargs):
        """ Synchronize local environment state with another environment object (base/main).

        Args:
            new_env (gym.Env): Source environment to be copied from.
        """
        if kwargs.get('verbose'):
            printGreen(
                '\nState of online learning environment is synched with global env.\n')
        self.env.__dict__ = source_env.__dict__.copy()

    def save_weights(self, save_dir):
        raise NotImplementedError

    def load_weights(self, save_dir):
        raise NotImplementedError


class RLS(Model):
    """
    Recursive Least Squares (RLS) incremental environment model
    """

    def __init__(self, config_dict: dict, env: gym.Env):
        super().__init__(config_dict, env)

        # Specific config passing
        self.gamma = self.config["gamma"]

        # Initialize covariance matrix
        self.Cov0 = self.config["cov0"] * \
            np.identity(self.state_size + self.action_size)

        # Low pass constant:
        self.tau = 0.005

        # Initial innovation (prediction error)
        self.epsilon_thresh = np.ones(
            self.state_size) * self.config["eps_thresh"]
        self.Cov_reset = False
        self.ctrl_state_idx = [7, 6, 5]
        # Scale:
        self.scale = np.array([10, 20, 10, 100, 1, 1, 1, 1, 1, 2000, 1, 1])

    @property
    def F(self):
        return np.float32(self.Theta[:self.state_size, :].T)

    @property
    def G(self):
        return np.float32(self.Theta[self.state_size:, :].T)

    def reset(self):
        # Initialize measurement matrix
        self.X = np.ones((self.state_size + self.action_size, 1))

        # Initialise parameter matrix
        self.Theta = np.zeros(
            (self.state_size + self.action_size, self.state_size))

        # State vector:
        self.x: np.array = np.zeros(self.state_size)

        # Error vector and Covariance matrix:
        self.epsilon = np.zeros((1, self.state_size))
        self.Cov = self.Cov0

        # Time
        self.t = 0.

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """
        Update RLS parameters based on one state-action sample.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.
            next_state (np.ndarray): Next state.
        """
        # TODO: verify the scaling
        state = state[:self.state_size]
        next_state = next_state[:self.state_size]

        state = np.divide(state, self.scale[self.ctrl_state_idx])
        next_state = np.divide(next_state, self.scale[self.ctrl_state_idx])

        # Predict next state
        action = self.env.scale_action(action)
        next_state_pred = self.predict(state, action)
        # print(next_state_pred.shape)

        self.epsilon = (np.array(next_state)[np.newaxis].T - next_state_pred).T
        # print(self.epsilon.shape)

        # Intermediate computations
        CovX = self.Cov @ self.X
        XCov = self.X.T @ self.Cov
        gammaXCovX = self.gamma + XCov @ self.X

        # Update parameter matrix - soft update
        _theta = self.Theta + (CovX @ self.epsilon) / gammaXCovX
        self.Theta = self.tau * self.Theta + (1 - self.tau) * _theta

        # Update covariance matrix
        self.Cov = (self.Cov - (CovX @ XCov) / gammaXCovX) / self.gamma

        # Check if Cov needs reset
        if self.Cov_reset is False:
            if np.sum(np.greater(np.abs(self.epsilon), self.epsilon_thresh)) == 1:
                self.Cov_reset = True
                self.Cov = self.Cov0
        elif self.Cov_reset is True:
            if np.sum(np.greater(np.abs(self.epsilon), self.epsilon_thresh)) == 0:
                self.Cov_reset = False

        self.t += self.dt

    def predict(self, state: np.ndarray, action: np.ndarray):
        """
            Predict next state based on RLS model

            Args:
                state (np.ndarray): Current state.
                action (np.ndarray): Current action.

            Returns:
                np.ndarray: Predicted state.
        """
        state = state[:self.state_size]

        # Set measurement matrix
        self.X = self.get_X(state, action).T

        # Predict next state
        next_state_pred = (self.X.T @ self.Theta).T
        # TODO: predict the change in state and not the state
        # X = Delta[theta, phi, beta, elevator, aileron, rudder]

        return next_state_pred

    def save_weights(self, save_dir):
        """
        Save current weights
        """
        filename = 'rls_model.npz'
        save_path = os.path.join(save_dir, filename)
        np.savez(
            save_path,
            x=self.X,
            theta=self.Theta,
            cov=self.Cov,
            epsilon=self.epsilon,
        )

    def load_weights(self, save_dir):
        """
        Load weights
        """

        # Weights npz
        path = os.path.join(save_dir, "rls_model.npz")
        npzfile = np.load(path)

        # Load weights
        self.X = npzfile["x"]
        self.Theta = npzfile["theta"]
        self.Cov = npzfile["cov"]
        self.epsilon = npzfile["epsilon"]


##############################
# Gaussian Process
# ############################

config_dict = {
    "kernel": RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
    "seed": 42
}


class GaussianProcess(Model, GaussianProcessRegressor):
    def __init__(self, config_dict: dict, env: AircraftEnv):
        super().__init__(config_dict, env)

        self.kernel = config_dict["kernel"]
        self.random_seed = config_dict["seed"]

        self.t = 0.
        self.dt = self.env.dt
        # self.state_size = min(10, self.config['state_size'], env.n_obs_full)
        self.sk_model = GaussianProcessRegressor(
            kernel=self.kernel, random_state=self.random_seed)

        if self.state_size == 3:
            self.ctrl_state_idx = [7, 6, 5]
        elif self.state_size == 6:
            self.ctrl_state_idx = [7, 6, 5, 0, 1, 2]
        # Scale:
        self.scale = np.array([10, 20, 10, 100, 1, 1, 1, 1, 1, 2000, 1, 1])

    def reset(self):
        pass

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """ Update model. GPR based on one state-action sample.
            Args:
                state (np.ndarray): State vector.
                action (np.ndarray): Action vector.
                next_state (np.ndarray): Next state vector."""

        # print(self.sk_model.kernel)
        self.kernel.set_params(**self.sk_model.kernel.get_params())
        self.sk_model = GaussianProcessRegressor(
            kernel=self.kernel, random_state=self.random_seed)

        state = state[:self.state_size]
        next_state = next_state[:self.state_size]

        state = np.divide(state, self.scale[self.ctrl_state_idx])
        next_state = np.divide(next_state, self.scale[self.ctrl_state_idx])

        action = self.env.scale_action(action)
        X = self.get_X(state, action)
        y = next_state.reshape(-1, self.state_size)
        self.sk_model.fit(X, y)

        self.t += self.dt

    def predict(self, state: np.ndarray, action: np.ndarray):
        """ Predict the next state based on GPR Model. """
        state = state[:self.state_size]
        action = self.env.scale_action(action)
        X = self.get_X(state, action)
        y_pred, std_pred = self.sk_model.predict(X, return_std=True)
        # printGreen(">> std pred: " + str(std_pred))
        return y_pred


class TransformerModel(nn.Module):
    def __init__(self, config: dict, env: AircraftEnv):
        super(TransformerModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.encoder = nn.Transformer(
            d_model=self.hidden_size, nhead=2, num_encoder_layers=2)
        self.decoder = nn.Transformer(
            d_model=self.hidden_size, nhead=2, num_decoder_layers=2)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.t = 0.
        self.dt = self.env.dt
        if self.state_size == 3:
            self.ctrl_state_idx = [7, 6, 5]
        elif self.state_size == 6:
            self.ctrl_state_idx = [7, 6, 5, 0, 1, 2]
        # Scale:
        self.scale = np.array([10, 20, 10, 100, 1, 1, 1, 1, 1, 2000, 1, 1])

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        pass

    def predict(self, state: np.ndarray, action: np.ndarray):
        pass


class NN(nn.Module):
    def __init__(self, args: ESParameters) -> None:
        super(NN, self).__init__()
        self.args = args

        self.seed = torch.manual_seed(args.seed)

        h = self.args.fdi_hidden_size
        self.L = self.args.fdi_num_layers
        self.activation = activations[args.activation_actor.lower()]

        layers = []
        layers.extend([
            nn.Linear(args.state_dim+args.action_dim, h),
            self.activation
        ])

        # hidden layers:
        for _ in range(self.L):
            layers.extend([
                nn.Linear(h, h),
                LayerNorm(h),
                self.activation
            ])

        # output layer:
        layers.extend([
            nn.Linear(h, args.fdi_out_size),
        ])
        # # print(*layers)
        self.net = nn.Sequential(*layers)
        # self.batchnn = nn.BatchNorm1d(self.args.state_dim+self.args.action_dim)
        self.to(args.device)

    def forward(self, state, action):
        if isinstance(state, np.ndarray) and isinstance(action, np.ndarray):
            nstate = torch.FloatTensor(copy.copy(state)).to(self.args.device)
            naction = torch.FloatTensor(copy.copy(action)).to(self.args.device)
        inp = torch.cat((nstate, naction), dim=-1)
        out1 = self.net(inp)
        return out1

    def extract_parameters(self):
        """ Extract the current flattened neural network weights """
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count+sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    def inject_parameters(self, pvec):
        """ Inject a flat vector of ANN parameters into the model's current neural network weights """
        count = 0
        pvec = torch.tensor(pvec)
        for name, param in self.named_parameters():
            # only alter W -- skip norms and biases:
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count+sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    def count_parameters(self):
        """ Number of parameters in the model """
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


class ESModelIdentification(object):
    def __init__(self, env: AircraftEnv, config: ESParameters):
        self.args = config
        self.env = env
        self.dt = self.env.dt
        self.t = 0
        self.fdi = NN(self.args)
        num_params = self.fdi.count_parameters()
        self.ESModel = CEM(
            num_params=num_params,
            params=config,
            sigma_init=config.sigma_init,
            mu_init=copy.deepcopy(
                self.fdi.extract_parameters().cpu().numpy()),
            pop_size=config.fdi_pop_size,
            antithetic=not config.fdi_pop_size % 2,
            parents=config.fdi_parents,
            elitism=config.elitism,
            adaptation=config.cem_with_adapt,
        )
        self.ctrl_state_idx = [7, 6, 5]

    def reset(self):
        pass

    def update(self, state, action, next_state):
        es_fdi = self.ESModel.ask(self.args.fdi_pop_size)
        # p = mp.Pool()
        # fits = p.map(partial(eval_theta, args=self.args, state=state,
        #              action=action, next_state=next_state), es_fdi)
        fits = ray.get([eval_theta_remote.remote(theta, self.args,
                        state, action, next_state) for theta in es_fdi])
        fits = np.array(fits)
        # print("Update: ", fits)
        self.ESModel.tell(es_fdi, fits)
        test_mu = eval_theta(self.ESModel.mu, self.args,
                             state, action, next_state)
        if test_mu > self.ESModel.best_mu_so_far_score:
            self.ESModel.best_mu_so_far = copy.deepcopy(self.ESModel.mu)
            self.ESModel.best_mu_so_far_score = test_mu
            # printPurple('>> New best mu so far: {}'.format(test_mu))
        self.fdi.inject_parameters(self.ESModel.elite)

    def predict(self, state, action):

        pred_next_state = self.fdi(
            state, action).detach().cpu().numpy()
        return pred_next_state

    def step(self, action):
        """ Simulate a step in the local environment. """
        is_done = False

        action = self.env.scale_action(action)

        # **** CONTROLLED STATE

        # [theta, phi, beta]=[7, 6, 5]

        # x = copy.copy(self.env.x[self.ctrl_state_idx])
        # x = np.divide(x, self.scale[self.ctrl_state_idx])

        obs = copy.copy(self.env.obs[:])
        pred_next_obs = self.predict(obs, action).flatten()
        # pred_next_obs = np.multiply(
        #     pred_next_obs, self.scale[self.ctrl_state_idx])
        self.env.x[self.ctrl_state_idx] += pred_next_obs[:3]
        self.env.x[[0, 1, 2]] = pred_next_obs[3:]  # p, q,r

        # Reward using clipped error
        reward = self.env.get_reward()

        # Calculate cost:
        cost = self.env.get_cost()

        # Update observation based on perfect observations & actuator state
        self.env.obs = np.hstack(
            (self.env.error, self.env.x[self.env.obs_idx]))
        self.env.last_u = action

        # Check for bounds and addd corresponding penalty for dying early
        is_done, penalty = self.env.check_bounds()
        reward += (0.2*penalty)

        # Step time
        self.env.t += self.env.dt
        self.t += self.env.dt
        self.env.last_obs = self.env.obs

        # info:
        info = {
            "ref": self.env.ref,
            "x":   self.env.x,
            "t":   self.env.t,
            "cost": cost,
        }
        return self.env.obs, reward, is_done, info

    def predictive_control(self, controller: object, t_horizon: float = 5.0, **kwargs):
        """ Predicts the trajectory of a controller (i.e., actor)
            for a finite horizon in the future.

            Args:
                controller (object): Actor object to be used during prediction.
                t_horizon (float, optional): Prediction horizon. Defaults to 5.

            Returns:
                tuple: List of rewards, list of time-steps played.
        """
        t_hor = t_horizon
        done = False
        rewards, action_lst, times = [], [], []

        # Initialise local progress:
        imagine_obs = self.env.obs
        t_start = self.env.t
        controller.eval()
        # play the future time-horizon
        while self.env.t < t_start + t_hor and not done:

            # select action:
            action = controller.select_action(imagine_obs)

            # Simulate one step in environment
            action = np.clip(action, -1, 1)

            # step
            next_obs, reward, done, _ = self.step(action.flatten())
            imagine_obs = next_obs

            # save
            rewards.append(reward)
            action_lst.append(self.env.last_u)
            times.append(self.env.t)

        rewards = np.asarray(rewards)
        action_lst = np.asarray(action_lst)
        times = np.asarray(times)

        return rewards, action_lst, times

    def sync_env(self, source_env: gym.Env, **kwargs):
        """ Synchronize local environment state with another environment object (base/main).

        Args:
            new_env (gym.Env): Source environment to be copied from.
        """
        if kwargs.get('verbose'):
            printGreen(
                '\nState of online learning environment is synched with global env.\n')
        self.env.__dict__ = source_env.__dict__.copy()


def eval_theta(theta, args, state, action, next_state):
    fdi = NN(args)
    fdi.inject_parameters(theta)
    pred_next_state = fdi(state, action).detach().cpu().numpy()
    rew = -np.sum(np.abs(pred_next_state - next_state))
    return rew


@ray.remote
def eval_theta_remote(theta, args, state, action, next_state):
    return eval_theta(theta, args, state, action, next_state)
