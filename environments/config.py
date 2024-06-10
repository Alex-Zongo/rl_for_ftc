import numpy as np
from environments.aircraftenv import AircraftEnv
try:
    from environments.lunarlander import LunarLanderWrapper
except:
    print("LunarLanderWrapper not available")


def select_env(environment_name: str, render_mode: bool = False, realtime: bool = False, use_state_history: bool = False, conform_with_sb: bool = False, add_sm_to_reward: bool = False, use_scaled_obs: bool = False, without_beta: bool = False):
    """
    This function returns the selected environment based on the following configuration:

     args:
        - environment_name: default str. the name of the environment
        - render_mode: default False - NotImplemented
        - realtime: default False - Make it True if render_mode is implemented
        - use_state_history: default False - if the agent does not use the state history
        - conform_with_sb: default True - to make sure the environment configuration as in Stable Baseline
        - add_sm_to_reward: default is False. to study the effect of smoothness metric on the reward
        - use_scaled_obs: default is False. to scale to states and observe the effects on training
        - without_beta: default False. to train the agent without `beta` part of the states during training.

    return:
        - Gym Wrapped Environment
    """
    _name = environment_name.lower()

    if 'lunar' in _name:
        wrapper = LunarLanderWrapper()
        return wrapper.env

    elif 'ph' in _name:
        tokens = _name.split('_')
        phlab_mode = 'nominal'
        if len(tokens) == 3:
            _, phlab_config, phlab_mode = tokens
        else:
            phlab_config = tokens[-1]
            phlab_mode = ""

        return AircraftEnv(
            configuration=phlab_config, mode=phlab_mode, render_mode=render_mode, realtime=realtime, use_state_history=use_state_history, conform_with_sb=conform_with_sb, add_sm_to_reward=add_sm_to_reward,
            use_scaled_obs=use_scaled_obs,
            without_beta=without_beta
        )
    else:
        raise ValueError(f"Unknown environment name: {environment_name}")


if __name__ == '__main__':
    env = select_env('Phlab_fullControl_nominal')
    env.reset()
    print(env.step(np.array([0.1, 0.1, 0.05])))
