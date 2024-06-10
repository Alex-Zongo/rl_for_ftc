from scipy.fftpack import fft
from abc import ABC, abstractmethod
# import pprint
from typing import List
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
# import jsbsim as jsb
from signals.base_signal import BaseSignal, Const
from signals.stochastic_signals import RandomizedCosineStepSequence
# import time

# Utility functions for printing
def printRed(skk): print(f"\033[91m {skk}\033[00m")
def printGreen(skk): print(f"\033[92m {skk}\033[00m")
def printLightPurple(skk): print(f"\033[94m {skk}\033[00m")
def printPurple(skk): print(f"\033[95m {skk}\033[00m")
def printCyan(skk): print(f"\033[96m {skk}\033[00m")
def printYellow(skk): print(f"\033[93m {skk}\033[00m")


class BaseEnv(gym.Env, ABC):
    """
    Base class for all environments.
    For the purpose of writing generic training, rendering and code that applies to all the Citation environments.
    """

    @property
    @abstractmethod
    def action_space(self) -> Box:
        """
        Returns the action space of the environment
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        """
        Returns the observation space of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def calc_reference_value(self) -> List[float]:
        """
        Returns the reference value of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def get_controlled_state(self) -> List[float]:
        """
        Returns the list of controlled states of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self) -> float:
        """
        Returns the reward of the environment
        """
        pass

    @abstractmethod
    def calc_error(self):
        """
        Returns the error of the environment
        """
        pass


class AircraftEnv(BaseEnv):
    """ Wrapper Environment for the Fault Environment Cases"""

    n_actions_full: int = 10
    n_obs_full: int = 12
    t: float = 0.
    dt: np.float16 = 0.01
    mToFeet: float = 3.28084
    g = 9.80665

    def __init__(self, configuration: str = None, mode: str = "nominal", render_mode: bool = False, realtime: bool = False, use_state_history: bool = False, conform_with_sb: bool = False, add_sm_to_reward: bool = False, use_scaled_obs: bool = False, without_beta=False, **kwargs):
        """
        args:
            - configuration: str - "attitude" to only control the aircraft attitude / else full state control.
            - mode: str - default "nominal" for normal flight case. e.g: "be" for partial loss of elevator.
            - render_mode: default False. NotImplemented (render via JSBSim for flightGear)
            - realtime: default False. Only if render_mode is effective.
            - use_state_history: default False. True if including state history in the agent training.
            - conform_with_sb: default False. to conform with Stable Baselines environment configuration.
            - add_sm_reward: default False. to study the effects of the smoothness metric on the reward.
            - use_scaled_obs: default False. to scale the states and investigate the effects.
            - without_beta: default False. Whether to include `beta` in the state.
            -
        """
        self.without_beta = without_beta
        self.render_mode = render_mode
        self.use_state_history = use_state_history
        self.conform_with_sb = conform_with_sb
        self.add_sm_to_reward = add_sm_to_reward
        self.use_scaled_obs = use_scaled_obs
        # self.enable_low_pass_filter = False
        # if self.enable_low_pass_filter:
        #     printYellow("Low pass filter enabled")
        self.action_history_for_filtering = None
        self.step_count = None
        self.ail_max = None
        self.el_max = None
        self.rud_stuck = None
        if self.add_sm_to_reward:
            printYellow("Adding smoothness to reward")

        # full state control
        # ******* Determine the action space and observation space ********
        self.n_actions = 3  # aileron da, elevator de, rudder dr
        if "attitude" in configuration.lower():
            # p, q, r, and alpha -> might change that to [phi, theta, psi, p, q, r] #TODO
            # print("Attitude Control.\n")
            # self.obs_idx = [0, 1, 2, 4] #
            self.obs_idx = [0, 1, 2]  # 3 is for Vtas

        else:
            print("Full State Control.\n")
            self.obs_idx = range(10)  # 10 states, all states

        # ***************************************************
        # if render_mode is implemented to link with jSBSim.
        # if self.render_mode:
        #     self.fdm = jsb.FGFDMExec("environments/JSBSim", None)
        #     self.fdm.load_model("citation")
        #     self.fdm.set_dt(self.dt)
        #     self.physicsPerSecond = int(1.0/self.dt)
        #     self.pauseDelay = 0.1  # how long an action is applied
        #     self.fdm.set_output_directive("data_output/flightgear.xml")
        #     self.fdm.load_ic("cruise_init", True)
        #     self.fdm.run_ic()
        #     self.fdm["gear/gear-cmd-norm"] = 0
        #     for _ in range(10):
        #         self.fdm.run()
        #     self.fdm.do_trim(1)
        #     self.fdm.print_simulation_configuration()
        #     self.realtime = realtime

        # ******* environment mode selection  ********
        if mode == "nominal" or 'h2000-v90' == mode.lower() or 'incremental' in mode.lower():
            from .h2000_v90 import citation as citation_h2000_v90
            self.aircraft = citation_h2000_v90
            # printGreen('Trim mode: h=2000 m v=90 m/s (nominal)')
        elif 'h2000-v150' == mode.lower() or 'high-q' == mode.lower():
            from .h2000_v150 import citation as citation_h2000_v150
            printCyan('Trim mode: h=2000 m v=150 m/s (high-q)')
            self.aircraft = citation_h2000_v150
        elif 'h10000-v90' == mode.lower() or 'low-q' == mode.lower():
            from .h10000_v90 import citation as citation_h10000_v90
            printCyan('Trim mode: h=10000 m v=90 m/s (low-q)')
            self.aircraft = citation_h10000_v90
        elif 'be' == mode.lower():  # be = broken elevator:
            from .be import citation as citation_be
            printRed('Trim mode: h-2000 m v=90 m/s Broken Elevator 70%')
            self.aircraft = citation_be
        elif 'jr' == mode.lower():  # jr = jammed rudder:
            from .jr import citation as citation_jr
            printRed('Trim mode: h-2000 m v=90 m/s Jammed Rudder at 15 deg')
            self.rud_stuck = np.deg2rad(15.0)
            self.aircraft = citation_jr
        elif 'se' == mode.lower():  # se = saturated elevator:
            from .se import citation as citation_se
            printRed(
                'Trim mode: h-2000 m v=90 m/s Saturated Elevator at +/- 2.5 deg')
            self.el_max = np.deg2rad(2.5)
            self.aircraft = citation_se
        elif 'sa' == mode.lower():  # sa = saturated aileron:
            from .sa import citation as citation_sa
            printRed(
                'Trim mode: h-2000 m v=90 m/s Saturated Aileron at +/- 1.0 deg')
            self.ail_max = np.deg2rad(1.0)
            self.aircraft = citation_sa
        elif 'ice' == mode.lower():  # ice = icing:
            from .ice import citation as citation_ice
            printRed('Trim mode: h-2000 m v=90 m/s Iced')
            self.aircraft = citation_ice
        elif 'noise' == mode.lower():  # noise = noise:
            from .noise import citation as citation_noise
            printRed('Trim mode: h-2000 m v=90 m/s Noisy sensors')
            self.aircraft = citation_noise
        elif 'cg-for' == mode.lower():  # cg-for = cg forward:
            from .cg_for import citation as citation_cg_for
            printRed('Trim mode: h-2000 m v=90 m/s CG Forward')
            self.aircraft = citation_cg_for
        elif 'cg' == mode.lower():  # cg = cg aft:
            from .cg import citation as citation_cg
            printRed('Trim mode: h-2000 m v=90 m/s CG Aft')
            self.aircraft = citation_cg
        elif 'cg-shift' == mode.lower():  # cg aft after 20s
            from .cg_timed import citation as citation_cg_timed
            printRed('Trim mode: h-2000 m v=90 m/s CG Aft after 20s')
            self.aircraft = citation_cg_timed
        elif 'gust' == mode.lower():  # gust = gust:
            from .gust import citation as citation_gust
            printRed('Trim mode: h-2000 m v=90 m/s Vertical Gust of 15ft/s at 20s')
            self.aircraft = citation_gust
        elif 'test' in mode.lower():  # test = test:
            from .test import citation as citation_test
            printRed('Test Case')
            self.aircraft = citation_test
        else:
            raise ValueError("Unknown trim condition or control mode")
        # ***********************************************

        # Whether to use incremental control.
        self.use_incremental = 'incremental' in mode.lower()
        if self.use_incremental:
            print('Incremental Control.')

        # Evaluation mode
        self.eval_mode: bool = False
        self.t_max: float = 20  # [s] to be changed

        # DASMAT Inputs ---> DASMAT States
        """
        0: de                       0: p
        1: da                       1: q
        2: dr                       2: r
        3: de trim                  3: V
        4: da trim                  4: alpha
        5: dr trim         --->     5: beta
        6: df                       6: phi
        7: gear                     7: theta
        8: throttle1                8: psi
        9: throttle2                9: he
                                   10: xe
                                   11: ye
        """
        self.x: np.ndarray = None  # observed state vector
        self.obs: np.ndarray = None
        self.last_obs: np.ndarray = None
        self.before_last_obs: np.ndarray = None
        self.V0: float = None  # [m/s]
        self.last_u: np.ndarray = None  # last input action or control
        self.before_last_u: np.ndarray = None

        # references to track
        self.ref: List[BaseSignal] = None
        self.ref_values: np.ndarray = None
        self.theta_trim: float = 0.22  # standard theta trim in degree

        # actuator bounds
        if self.use_incremental:
            self.bound = np.deg2rad(25)  # [deg/s]
        else:
            self.bound = np.deg2rad(10)  # [deg]

        # state bounds
        self.max_theta = np.deg2rad(60.0)  # [deg]
        self.max_phi = np.deg2rad(75.0)  # [deg]
        self.n = self.n_actions-1 if self.without_beta else self.n_actions

        if self.use_incremental:
            # aircraft state + actuator state + control states error (equal size with actuator states)
            self.n_obs: int = len(self.obs_idx) + 2*self.n
        else:
            # aircraft state + control states error
            self.n_obs: int = len(self.obs_idx) + self.n

        self.error: np.ndarray = np.zeros((self.n))
        # state error initialization
        if self.without_beta:
            self.error_scaler = np.array([1.0, 1.0]) * 6/np.pi
        else:
            # error scaler
            self.error_scaler = np.array([1.0, 1.0, 4.0]) * 6/np.pi

        self.error_scaler = self.error_scaler[:self.n]

        self.max_bound = np.ones(self.error.shape)  # bounds for state error
        self.min_bound = np.zeros(self.error.shape)

    @property
    def action_space(self):
        """actuators bounds in radians"""
        if self.conform_with_sb:
            return Box(
                low=-np.ones(self.n_actions),
                high=np.ones(self.n_actions),
                dtype=np.float32,  # avoid cast errors
            )
        return Box(
            low=-self.bound*np.ones(self.n_actions),
            high=self.bound*np.ones(self.n_actions),
            dtype=np.float64,
            # dtype=np.float32,  # avoid cast errors
        )

    # def filter_control_input(self, deflection):
    #     w_0 = 2*2*np.pi  # rad/s
    #     filtered_deflection = deflection.copy()
    #     if self.step_count > 1 and self.enable_low_pass_filter:
    #         # filtered_deflection = 0.5*deflection + 0.5*filtered_deflection
    #         # filtered_deflection = filtered_deflection + (self.dt*w_0)/(1 + self.dt*w_0)*(deflection - filtered_deflection)
    #         filtered_deflection = self.action_history_for_filtering[:, self.step_count-1] / (
    #             1+w_0*self.dt) + deflection * (w_0*self.dt) / (1+w_0*self.dt)
    #     return filtered_deflection

    def scale_action(self, action: np.array):
        """
        Returns the scaled action of the environment from the clipped action [-1,1] to [action_space.low, action_space.high]
        """
        if self.conform_with_sb:
            low, high = self.bound*self.action_space.low, self.bound*self.action_space.high
            if self.ail_max:
                low[1], high[1] = -self.ail_max, self.ail_max
            if self.el_max:
                low[0], high[0] = -self.el_max, self.el_max
            if self.rud_stuck:
                low[2], high[2] = self.rud_stuck - \
                    np.deg2rad(1), self.rud_stuck+np.deg2rad(1)

            # print(low, high)
        else:
            low, high = self.action_space.low, self.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)

    def unscale_action(self, action: np.array):
        """
        Rescale the action from [action_space.low, action_space.high] to [-1,1]
        """
        if self.conform_with_sb:
            low, high = self.bound*self.action_space.low, self.bound*self.action_space.high
        else:
            low, high = self.action_space.low, self.action_space.high
        return 2.0 * (action - low) / (high - low) - 1

    @property
    def observation_space(self):
        """Return states bounds in degrees (phi, theta, psi, ...)"""
        # if self.use_state_history:
        #     return Box(
        #         low=-30*np.ones(self.n_obs*2),
        #         high=30*np.ones(self.n_obs*2),
        #         dtype=np.float64,
        #     )
        return Box(
            low=-30*np.ones(self.n_obs),
            high=30*np.ones(self.n_obs),
            dtype=np.float64,
        )

    @property
    def p(self):
        """p: is the roll rate in rad/s"""
        return self.x[0]

    @property
    def q(self):
        """q: is the pitch rate in rad/s"""
        return self.x[1]

    @property
    def r(self):
        """r: is the yaw rate in rad/s"""
        return self.x[2]

    @property
    def V(self):
        """v: is the airspeed in m/s"""
        return self.x[3]

    @property
    def alpha(self):
        """ alpha is the angle of attack in radians"""
        return self.x[4]

    @property
    def beta(self):
        """ beta is the sideslip angle radians"""
        return self.x[5]

    @property
    def phi(self):
        """ phi is the roll angle in radians"""
        return self.x[6]

    @property
    def theta(self):
        """ theta is the pitch angle in radians"""
        return self.x[7]

    @property
    def psi(self):
        """ psi is the yaw angle in radians"""
        return self.x[8]

    @property
    def h(self):
        """ h is the altitude in meters"""
        return self.x[9]

    @property
    def nz(self):
        """ nz is the load factor"""
        return 1.0 + self.q * self.V / (self.g)

    def set_eval_mode(self, t_max: int = 80):
        """Switch to Evaluation Mode"""
        self.t_max = t_max
        self.eval_mode = True
        # if user_eval_refs is not None: self.user_refs = user_eval_refs
        printYellow(
            f"Switching to evaluation mode:\n Tmax = {self.t_max} seconds \n")

    def init_ref(self, **kwargs):
        """
        Initializes the reference signals.
        Assuming n_actions: 3
        """
        step_beta = Const(0.0, self.t_max, 0.0)

        # refs signals for phi and theta
        if "user_refs" not in kwargs:
            self.theta_trim = np.rad2deg(self.theta)
            step_theta = RandomizedCosineStepSequence(
                t_max=self.t_max,
                ampl_max=30,
                block_width=self.t_max//5,
                smooth_width=self.t_max//6,
                n_levels=self.t_max//2,
                vary_timings=self.t_max/500.
            )

            step_phi = RandomizedCosineStepSequence(
                t_max=self.t_max,
                ampl_max=20,
                block_width=self.t_max//5,
                smooth_width=self.t_max//6,
                n_levels=self.t_max//2,
                vary_timings=self.t_max/500.
            )
        else:
            if not self.eval_mode:
                Warning(
                    "user reference signals have been given while env is not in evaluation mode")
            step_theta = kwargs["user_refs"]['theta_ref']
            step_phi = kwargs["user_refs"]['phi_ref']

        step_theta += Const(0.0, self.t_max, self.theta_trim)
        if self.without_beta:
            self.ref = [step_theta, step_phi]
        else:
            self.ref = [step_theta, step_phi, step_beta]

        # print(self.ref)

    def calc_reference_value(self):
        """ Convert the reference signal from degree to radians"""

        # Calculates the reference value for the current time step (theta, phi, psi)
        self.ref_values = np.asarray(
            [np.deg2rad(ref_signal(self.t)) for ref_signal in self.ref])

    def get_controlled_state(self):
        """ Returns the values of the controlled states """
        if self.without_beta:
            ctrl = ctrl = np.asarray(
                [self.theta, self.phi])  # without beta
        else:
            ctrl = np.asarray([self.theta, self.phi, self.beta]
                              )  # replaced beta with psi

        # return ctrl[:self.n_actions]
        return ctrl[:self.n]

    def calc_error(self):
        """ Calculates the error between the controlled states and the reference values """
        self.calc_reference_value()

        self.error[:self.n] = self.ref_values - \
            self.get_controlled_state()

    def action_smoothness(self, actions_history: np.ndarray):
        """ Calculates the smoothness of from t-2 to t action history """
        N, A = actions_history.shape
        T = N * self.dt
        freq = np.linspace(self.dt, 1/(2*self.dt), N)
        Syy = np.zeros((N, A))
        for i in range(A):
            Y = fft(actions_history[:, i], N)
            Syy_disc = Y[0:N]*np.conjugate(Y[0:N])
            Syy[:, i] = np.abs(Syy_disc) * self.dt
        signal_roughness = np.einsum('ij,i->j', Syy, freq) / N
        _S = np.sqrt(np.sum(signal_roughness))
        roughness = _S / 100  # _S * 0.03 / T
        return -roughness

    def average_absolute_diff(self, actions_history):
        """ Average Absolute difference: A smoothness computation strategy to evaluate the roughness of actions history."""

        assert actions_history.shape[0] == 3, "actions_history should have 3 rows"
        a1 = np.abs(actions_history[0, :]-actions_history[1, :])
        a2 = np.abs(actions_history[1, :]-actions_history[2, :])
        return -np.mean((a1+a2)/2)/10

    def maximum_diff(self, actions_history):
        """ Maximum Difference: A smoothness computation strategy to evaluate the roughness of actions history."""

        a1 = np.abs(actions_history[0, :]-actions_history[1, :])
        a2 = np.abs(actions_history[1, :]-actions_history[2, :])
        s_max = np.max([a1, a2], axis=0)
        return -np.mean(s_max)/10

    def variance_of_diff(self, actions_history):
        """ Variance of Difference: A smoothness computation strategy to evaluate the roughness of actions history."""

        assert actions_history.shape[0] == 3, "actions_history should have 3 rows"

        a1 = np.abs(actions_history[0, :]-actions_history[1, :])
        a2 = np.abs(actions_history[1, :]-actions_history[2, :])
        d_bar = (a1+a2)/2

        s = ((a1-d_bar)**2 + (a2-d_bar)**2)/2
        return -np.mean(s)/10

    def get_reward(self):
        """ Calculates the reward """
        self.calc_error()
        rate_penalty = -np.sum(np.abs(self.x[0:3]))/3
        angle_reward = -np.sum(np.abs(np.clip(self.error * self.error_scaler, -
                                              self.max_bound, self.max_bound)))/self.error.shape[0]

        weights = [0.6, 0.2]
        reward = weights[0]*angle_reward + weights[1]*rate_penalty

        return float(reward)

    def get_cost(self):
        """ the binary cost of  the last transition -> defining good behavior of the airplane"""

        if np.rad2deg(np.abs(self.alpha)) > 11.0 or \
           np.rad2deg(np.abs(self.phi)) > 0.75 * self.max_phi or \
           self.V < self.V0/3:
            return 1
        return 0

    def incremental_control(self, action: np.ndarray):
        """ low-pass filtered Incremental control input for the citation model """
        return self.last_u + action * self.dt

    def pad_action(self, action: np.ndarray):
        """ Pad action with zeros to correspond to the simulink model input dimensions"""
        # TODO might modify the control inputs to include throttle cmd for a full body control
        citation_input = np.pad(
            action, (0, self.n_actions_full - self.n_actions), 'constant', constant_values=(0.0,))
        return citation_input

    def check_bounds(self):
        """ Additional penalty for exceeding the bounds"""
        if self.t >= self.t_max \
                or np.abs(self.theta) > self.max_theta \
                or np.abs(self.phi) > self.max_phi \
                or self.h < 50:
            # negative reward for dying soon
            penalty = -1/self.dt * (self.t_max - self.t) * 2
            return True, penalty
        return False, 0.

    def reset(self, seed=0, **kwargs):
        """ Reset the env to initial conditions """

        self.t = 0.0  # reset time
        self.np_random = np.random.seed(seed)
        self.aircraft.initialize()  # initialize the aircraft simulink model

        # initial input step (full-zero input to retrieve retrive the states)
        self.last_u = np.zeros(self.n_actions)
        self.last_action = np.zeros(self.n_actions)
        self.before_last_action = np.zeros(self.n_actions)

        # self.step_count = 0
        # self.action_history_for_filtering = np.zeros(
        #     (self.action_space.shape[0], int(self.t_max//self.dt)+2))

        act_sm = 0
        if self.add_sm_to_reward:
            self.action_history_for_fft = [self.last_u]
            self.action_history = np.vstack(
                (self.before_last_action, self.last_action, self.last_u))
            self.act_sm = 0
            act_sm = self.act_sm

        # init state vector after padding the input
        _input = self.pad_action(self.last_u)
        self.x = self.aircraft.step(_input)

        # init aircraft reference conditions and randomized reference signal sequence
        self.V0 = self.V
        self.init_ref(**kwargs)

        # the observed state
        if self.use_scaled_obs:
            error = self.error_scaler * self.error
        else:
            error = self.error
        self.obs = np.hstack((error.flatten(), self.x[self.obs_idx]))
        self.last_obs = self.obs[:]
        self.before_last_obs = self.obs[:]

        if self.use_incremental:
            self.obs = np.hstack((self.obs, self.last_u))

        if self.use_state_history:
            obs = np.vstack((self.obs, self.last_obs, self.before_last_obs))
        else:
            obs = self.obs
        # if self.render_mode:
        #     self.fdm.load_ic("cruise_init", True)
        #     self.fdm.run_ic()
        #     self.fdm.do_trim(1)

        # info:
        info = {
            "ref": self.ref,
            "x": self.x,
            "t": self.t,
            "sm": act_sm,
            "sm_fft": act_sm,
        }

        return obs, info

    def activate_render_mode(self):
        if self.render_mode:
            self.fdm.load_ic("cruise_init", True)
            self.fdm.run_ic()
            self.fdm.do_trim(1)

    def step(self, action: np.ndarray):
        """ One Step in time by the agent in the environment
            Args:
                action: the action taken by the agent - Un-scaled input in the interval of [-1, 1]
            Returns:
                Tuple: new_state, the reward, is_done mask and info {reference signal value, time, cost of step}
        """
        is_done = False
        self.before_last_obs = self.last_obs
        self.last_obs = self.obs

        if self.add_sm_to_reward:
            self.action_history_for_fft.append(action)
            sm_fft = self.action_smoothness(
                np.asarray(self.action_history_for_fft))
            self.action_history = np.vstack(
                (self.before_last_action, self.last_action, action))
            # act_sm = self.action_smoothness(self.action_history)
            act_sm = self.maximum_diff(self.action_history)
            # print("act sm", act_sm)
            self.before_last_action = self.last_action
            self.last_action = action
            self.act_sm = act_sm

        else:
            act_sm = 0.0
            sm_fft = 0.0

        # print(f"SM: {act_sm}")
        # scale the action to the actuator rate bounds
        action = self.scale_action(action)  # rad

        if self.use_incremental:
            u = self.incremental_control(action)
        else:
            u = action
        # rendering
        # if self.render_mode:
        #     self.render(u)

        # pad the input action to match the dimensions of the simulink model
        # u_filtered = self.filter_control_input(u)
        _input = self.pad_action(u)
        self.x = self.aircraft.step(_input)

        # self.action_history_for_filtering[:, self.step_count] = u_filtered
        # self.step_count += 1
        # get the reward
        reward = self.get_reward()

        # cost:
        cost = self.get_cost()

        # update observation based on perfect observations and actuator state:
        # self.calc_error()  # update observation state error
        if self.use_scaled_obs:
            error = self.error * self.error_scaler
        else:
            error = self.error
        self.obs = np.hstack((error.flatten(), self.x[self.obs_idx]))

        self.last_u = u

        if self.use_incremental:
            self.obs = np.hstack((self.obs, self.last_u))

        if self.use_state_history:
            obs = np.vstack((self.obs, self.last_obs, self.before_last_obs))
        else:
            obs = self.obs

        # check the bounds and add corresponding penalty for terminated early:
        is_done, penalty = self.check_bounds()
        reward += (0.2*penalty)
        # print("reward :", reward)
        # reward += act_sm

        # step time:
        self.t += self.dt

        # info:
        info = {
            "ref": self.ref,
            "x": self.x,
            "t": self.t,
            "cost": cost,
            "sm": act_sm,
            "sm_fft": sm_fft
        }
        if self.conform_with_sb:
            truncated = is_done
            return obs, reward, is_done, truncated, info
        return obs, reward, is_done, info

    def finish(self):
        """ Terminate the simulink simulation"""
        self.aircraft.terminate()

    # def send_to_fg(self, action: np.ndarray):
    #     [de, da, dr] = action  # Un/or -scaled action in the interval of [-1, 1]
    #     # prt = dict(
    #     #     de=de,
    #     #     da=da,
    #     #     dr=dr,
    #     # )
    #     # printCyan(prt)
    #     # ******* rad position of the controls
    #     # self.fdm["fcs/elevator-pos-rad"] = de
    #     # self.fdm["fcs/left-aileron-pos-rad"] = da
    #     # self.fdm["fcs/right-aileron-pos-rad"] = -da
    #     # self.fdm["fcs/rudder-pos-rad"] = dr

    #     # ******* normalized position of the controls
    #     trim_de = self.fdm["fcs/pitch-trim-cmd-norm"]
    #     trim_da = self.fdm["fcs/roll-trim-cmd-norm"]
    #     trim_dr = self.fdm["fcs/yaw-trim-cmd-norm"]

    #     self.fdm['fcs/aileron-cmd-norm'] = da-trim_da
    #     self.fdm['fcs/elevator-cmd-norm'] = de-trim_de
    #     self.fdm['fcs/rudder-cmd-norm'] = dr-trim_dr

    # def render(self, action: np.ndarray):
    #     # printCyan("Rendering")
    #     # printCyan(f"Action Scaled de-da-dr  : {action}")

    #     u = self.unscale_action(action)
    #     # u = action  # scaled action
    #     # printCyan(f"Action Unscaled de-da-dr: {u}")

    #     self.send_to_fg(u)
    #     for _ in range(int(self.pauseDelay*self.physicsPerSecond)):
    #         self.send_to_fg(u)
    #         self.fdm.run()
    #         if self.realtime:
    #             time.sleep(self.dt)

    #     # self.print_rendered_action()

    # def print_rendered_action(self):
    #     de_cmd_sum = self.fdm["fcs/pitch-trim-sum"]
    #     da_cmd_sum = self.fdm["fcs/roll-trim-sum"]
    #     dr_cmd_sum = self.fdm["fcs/rudder-command-sum"]

    #     de = self.fdm["fcs/elevator-pos-rad"]
    #     da = self.fdm["fcs/left-aileron-pos-rad"]
    #     dr = self.fdm["fcs/rudder-pos-rad"]

    #     ap_de = self.fdm["ap/elevator_cmd"]
    #     ap_da = self.fdm["ap/aileron_cmd"]
    #     ap_dr = self.fdm["ap/rudder_cmd"]

    #     trim_de = self.fdm["fcs/pitch-trim-cmd-norm"]
    #     trim_da = self.fdm["fcs/roll-trim-cmd-norm"]
    #     trim_dr = self.fdm["fcs/yaw-trim-cmd-norm"]

    #     # printCyan(
    #     #     f"Action trim-cmd-sum: [{de_cmd_sum, da_cmd_sum, dr_cmd_sum}]")
    #     printCyan(f"Action actual delta(rad) : [{de, da, dr}]")
    #     # printCyan(f"Action trim-cmd: [{trim_de, trim_da, trim_dr}]")
