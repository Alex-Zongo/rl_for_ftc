import numpy as np
import typing
import gymnasium as gym


class LunarLanderWrapper(gym.Wrapper):
    env = gym.make('LunarLanderContinuous-v2', render_mode="human")

    def __init__(self, base_env=env):
        super().__init__(base_env)
        self.env = base_env

    def simulate(self,
                 actor: object,
                 render: bool = False,
                 broken_engine: bool = False,
                 state_noise: bool = False,
                 noise_intensity: float = 0.05):
        """
        Wrapper function for the gymnasium LunarLander environment.
        It include the following faulty case:
            * broken main engine
            * faulty navigation sensors: noisy position

        params:
            actor: object -> actor class that has the select_action() method

            env: object -> the Environment with OpenAi Gymnasium has the following methods (make, reset, step)

            render: bool -> whether to render the video of the simulation. default is False

            broken_engine: bool,  -> clip the main engine to 75% (fault cases for evaluation only). Defaults to False.

            state_noise: bool, optional -> add a zero-mean Gaussian noise to the observations (x, y) only for evaluation. default=False

            noise_intensity: float, optional -> standard deviation of the Gaussian Noise added to the state. Default to 0.005

        Returns:
            tuple: Reward: float, x-impact-position: float, y-impact-velocity: float
        """

        total_reward = 0.0
        x_impact_pos = None
        y_impact_vel = None
        all_y_vels = []

        # reset the env
        terminated = False
        state, _ = self.env.reset()

        while not terminated:
            if render:
                self.env.render()

            # Actor chooses the action
            action = actor.select_action(np.array(state))

            # simulate one step in the environment
            # fault -- main engine is clipped to 75% of normal operational behavior
            if broken_engine:
                action[0] = np.clip(action[0], -1, 0.5)

            # step into the environment
            next_state, reward, terminated, truncated, info = self.env.step(
                action.flatten())

            # update
            total_reward += reward
            state = next_state

            # fault -- noisy position estimation
            if state_noise:
                noise = np.random.normal(0, noise_intensity, 2)
                state[:2] = state[:2] + noise

            # Boundary characteristics:
            x_pos = state[0]
            y_vel = state[3]
            leg0_touch = bool(state[6])
            leg1_touch = bool(state[7])
            all_y_vels.append(y_vel)

            # check for first impact on the ground
            if x_impact_pos is None and (leg0_touch or leg1_touch):
                x_impact_pos = x_pos
                y_impact_vel = y_vel

        # if no impact edge case
        if x_impact_pos is None:
            x_impact_pos = x_pos
            y_impact_vel = min(all_y_vels)

        return {"total_reward": total_reward, 'behavioral_char': (x_impact_pos, y_impact_vel)}

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """ Scale the actions from [-1, 1] to the appropriate scale of the action space. """
        low, high = self.env.action_space.low, self.env.action_space.high
        action = low + 0.5 * (action + 1.0) * (high - low)
        return action
