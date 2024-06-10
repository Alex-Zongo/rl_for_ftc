from pathlib import Path
import os
import numpy as np
import torch
from core_algorithms.some_actor_model import LSTM_ACTOR, Actor, RNN_Actor
from core_algorithms.td3 import Actor as TD3_Actor
from dataclasses import dataclass
from signals.sequences import SmoothedStepSequence


@dataclass
class Stats:
    nmae: np.float16
    nmae_std: np.float16
    sm: np.float16
    sm_std: np.float16


def gen_refs(t_max: int, ampl_times: np.array, ampl_max: float, num_trials: int = 10):
    """Generate a list of reference smoothened step signals. from 0 to t_max.

    Args:
        t_max (int): Episode time.
        ampl_times (np.array): Starting times of each new step block.
        ampl_max (float): Maximum amplitude of the reference signal, symmetric wrt zero.
        num_trials (int, optional): number of random references. Defaults to 10.
    """
    refs_lst = []

    for _ in range(num_trials):
        # Possible choices:
        ampl_choices = np.linspace(-ampl_max, ampl_max, 6)

        # Generate random amplitudes:
        amplitudes = np.random.choice(ampl_choices, size=6, replace=True)
        amplitudes[0] = 0.0

        # disturb starting times for each step:
        ampl_times = [ampl_times[0]] + [
            t + np.random.uniform(-0.05, 0.05) for t in ampl_times[1:]
        ]

        # step object:
        _step = SmoothedStepSequence(
            times=ampl_times,
            amplitudes=amplitudes,
            smooth_width=t_max//10
        )
        refs_lst.append(_step)

    return refs_lst


def find_logs_path(logs_name: str, root_dir: str = './logs/wandb/'):
    cwd = os.getcwd()

    if not cwd.endswith('control'):
        pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
        cwd = pwd
    wandb = cwd / Path(root_dir)

    logs_name = logs_name.lower()
    for _path in wandb.iterdir():
        if _path.is_dir():
            if _path.stem.lower().endswith(logs_name):
                print(_path.stem.lower())
                return wandb / _path
    return None


# def load_pop(model_path: str, args):
#     """ Load evolutionary population"""
#     model_path = model_path / Path('files/')
#     actor_path = os.path.join(model_path, 'evolution_agents.pkl')

#     agents_pop = []
#     checkpoint = torch.load(actor_path)

#     for _, model in checkpoint.items():
#         _agent = GeneticAgent(args)
#         _agent.actor.load_state_dict(model)
#         agents_pop.append(_agent)

#     print("Genetic actors loaded from: " + str(actor_path))

#     return agents_pop



def load_agent_online(model_path: str, args, agent_type_arg):
    model_path = model_path / Path('files')
    if agent_type_arg == "use_mu":
        print('Using Mu agent')
        agent_name = 'mu_agent.pkl'
    elif agent_type_arg == "use_best_mu":
        print('Using Best Mu agent seen')
        agent_name = 'best_mu_agent.pkl'
    elif agent_type_arg == "use_best_elite":
        print('Using Best Elite agent seen')
        agent_name = 'best_elite_agent.pkl'
    else:
        print('Using Elite of last gen')
        agent_name = 'elite_agent.pkl'

    actor_path = os.path.join(model_path, agent_name)
    checkpoint = torch.load(actor_path)
    # agent = RNN_Actor(args, init=True) if args.use_rnn else Actor(
    #     args, init=True)
    agent = LSTM_ACTOR(args, args.device, args.state_dim, args.actor_hidden_size, args.actor_num_layers,
                       args.state_length) if args.use_state_history else TD3_Actor(args, init=True)
    agent.load_state_dict(checkpoint)

    print("Agent actor loaded from: " + str(actor_path))

    return agent


def load_agent(model_path: str, args, agent_type_arg):
    model_path = model_path / Path('files')
    if agent_type_arg.use_mu:
        print('Using Mu agent')
        agent_name = 'mu_agent.pkl'
    elif agent_type_arg.use_best_mu:
        print('Using Best Mu agent seen')
        agent_name = 'best_mu_agent.pkl'
    elif agent_type_arg.use_best_elite:
        print('Using Best Elite agent seen')
        agent_name = 'best_elite_agent.pkl'
    else:
        print('Using Elite of last gen')
        agent_name = 'elite_agent.pkl'

    actor_path = os.path.join(model_path, agent_name)
    checkpoint = torch.load(actor_path)
    # agent = RNN_Actor(args, init=True) if args.use_rnn else Actor(
    #     args, init=True)
    agent = LSTM_ACTOR(args, args.device, args.state_dim, args.actor_hidden_size, args.actor_num_layers,
                       args.state_length) if args.use_state_history else TD3_Actor(args, init=True)
    agent.load_state_dict(checkpoint)

    print("Agent actor loaded from: " + str(actor_path))

    return agent


def load_mu_agent(model_path: str, args):
    model_path = model_path / Path('files')
    actor_path = os.path.join(model_path, 'mu_agent.pkl')
    checkpoint = torch.load(actor_path)
    agent = RNN_Actor(args, init=True) if args.use_rnn else Actor(
        args, init=True)
    agent.load_state_dict(checkpoint)

    print("Mu actor loaded from: " + str(actor_path))

    return agent


def load_cov(model_path: str, verbose: bool = False):
    """ Load controller configuration from file.

    Args:
        model_path (str): Absolute path to logging folder.

    Returns:
        dict: Configuration dictionary.
    """
    model_path = model_path / Path('files/')
    cov_path = os.path.join(model_path, 'best_cov.csv')
    cov = np.loadtxt(cov_path, delimiter=',')
    return cov


def gen_eval_refs(amp_theta, amp_phi, max_theta, max_phi,  t_max, num_trails):
    """
    Generate evaluation references for the attitude control task.
    """
    t_length = len(amp_theta)
    time_array = np.linspace(0, t_max, t_length)
    base_ref_theta = SmoothedStepSequence(
        times=time_array, amplitudes=amp_theta, smooth_width=t_max//10)
    theta_refs = gen_refs(
        t_max=t_max,
        ampl_max=max_theta,
        ampl_times=time_array,
        num_trials=num_trails
    )
    theta_refs.append(base_ref_theta)

    base_ref_phi = SmoothedStepSequence(
        times=time_array, amplitudes=amp_phi, smooth_width=t_max//10)
    phi_refs = gen_refs(
        t_max=t_max,
        ampl_max=max_phi,
        ampl_times=time_array,
        num_trials=num_trails
    )
    phi_refs.append(base_ref_phi)

    user_eval_refs = list(zip(theta_refs, phi_refs))

    return user_eval_refs
