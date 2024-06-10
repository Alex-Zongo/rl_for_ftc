# for different seed:
# train the agent
# record the results

# average the results over the seeds:

# make sweep:
# GAUSSIAN Process with 10 sweep configuration:
# Consider one RL agent
# different sigma init values:
# different actor config:
# different learning rate:
# different critic config:

# LOOK FOR WAYS TO TWEAK THE FAULT PARAMETERS:
# COMBINED FAULTS AND EVALUATE:
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import time
import os
import argparse
import torch
from torch.optim import Adam
# from tqdm import tqdm
from core_algorithms.td3 import Actor, TD3ES
from core_algorithms.some_actor_model import CONV_ACTOR
from core_algorithms.sac import SACGaussianActor, SAC
from core_algorithms.multi_agent import MultiAgentActor
from core_algorithms.cem_rl import CEM
from core_algorithms.random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from core_algorithms.replay_memory import ReplayMemory
from core_algorithms.utils import fitness_function, train_rl, evaluate
from environments.aircraftenv import printPurple, printRed, printYellow
from environments.config import select_env
# from evaluation_es_utils import find_logs_path
from parameters_es import ESParameters


# *** SOME CONSTANTS ***#
SEED_RANGE = [0, 7, 42]
EnvIDs = ['NOMINAL', 'ICE', 'CG_SHIFT', 'SATURATED_AILERON', 'SATURATED_ELEVATOR',
          'BROKEN_ELEVATOR', 'JAMMED_RUDDER', 'CG_FOR', 'CG_AFT', 'HIGH_Q', 'LOW_Q', 'NOISE', 'GUST']
ENVS = dict(
    # LUNAR_LANDER='LunarLanderContinuous-v2',  # for quick tests

    NOMINAL='PHlab_attitude_nominal',
    ICE='PHlab_attitude_ice',
    CG_SHIFT='PHlab_attitude_cg-shift',  # cg shift aft after 20s
    SATURATED_AILERON='PHlab_attitude_sa',
    SATURATED_ELEVATOR='PHlab_attitude_se',
    BROKEN_ELEVATOR='PHlab_attitude_be',
    JAMMED_RUDDER='PHlab_attitude_jr',
    # CG_FOR='PHlab_attitude_cg-for',
    # CG_AFT='PHlab_attitude_cg',

    HIGH_Q='PHlab_attitude_high-q',
    LOW_Q='PHlab_attitude_low-q',
    NOISE='PHlab_attitude_noise',
    # GUST='PHlab_attitude_gust',
)

if __name__ == "__main__":
    # *********** ARGUMENTS PARSER **************#
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str,
                        default=ENVS['NOMINAL'], help='Environment name')
    parser.add_argument('--use-state-history', action='store_true',
                        help='Whether to use state history or not')
    parser.add_argument('--state-length', type=int, default=3)
    parser.add_argument('--seed', type=int,
                        default=SEED_RANGE[1], help='Random seed to be used')
    parser.add_argument('--start-steps', default=1000,
                        help='buffer size before training RL', type=int)
    parser.add_argument('--should-log', action='store_true',
                        help='Whether to log or not to WandB')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--cem-with-adapt', action='store_true',
                        help='whether to use CEM with sigma adaptation')

    parser.add_argument('--policy-noise', default=0.05, type=float)
    parser.add_argument('--noise-sd', type=float, default=0.1)
    parser.add_argument('--noise-clip', default=0.2, type=float)
    parser.add_argument('--policy-update-freq', default=2, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    # parser.add_argument('--cmb_size', default=10000, type=int)
    parser.add_argument('--actor-num-layers', default=2, type=int)
    parser.add_argument('--actor-hidden-size', default=96, type=int)
    parser.add_argument('--actor-lr', default=4e-4, type=float)
    parser.add_argument('--critic-num-layers', default=2, type=int)
    parser.add_argument('--critic-hidden-size',
                        default=[32, 32])
    parser.add_argument('--critic-lr', default=3e-4, type=float)

    # Gaussian noise parameters
    parser.add_argument('--gauss-sigma', default=0.05, type=float)

    # OU process parameters
    parser.add_argument('--ou-noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou-theta', default=0.15, type=float)
    parser.add_argument('--ou-sigma', default=0.2, type=float)
    parser.add_argument('--ou-mu', default=0.0, type=float)

    # ES parameters:
    parser.add_argument('--pop-size', default=20, type=int)
    parser.add_argument('--parents', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n-grad', default=0, type=int)
    parser.add_argument('--sigma-init', default=0.3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp-limit', default=1e-5, type=float)
    parser.add_argument('--mult-noise', dest='mult_noise', action='store_true')
    parser.add_argument('--weight-decay', default=0.01,
                        type=int, help='weight decay for fitness')
    parser.add_argument('--n-noisy', default=0, type=int)

   # Training parameters:
    parser.add_argument('--n-evals', default=2, type=int)
    parser.add_argument('--n-generations', default=2, type=int)

    parser.add_argument('--maxiter', default=50, type=int)
    # parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--sample-ratio', default=[0.8, 0.1, 0.1], type=list, help='for the batch selection use for training')
    parser.add_argument('--mem-size', default=200000000, type=int)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--use-sac', action='store_true')
    parser.add_argument('--use-multiAgent', action='store_true')
    parser.add_argument('--run-name', default="CEM_TD3Buffers_pop50_parents10", type=str, required=True)
    parser.add_argument('--wandb-project-name', type=str, help='WandB project name for logging', default='Fault_tolerant_flight_control_CEMTD3Buffers')
    parser.add_argument('--entity-name', type=str, help='WandB entity name', )

    args = parser.parse_args()

    parameters = ESParameters(conf=args, init=True)
    run_name = args.run_name
    # "CEM_TD3Buffers_MultiAgent_obsLength1_stateDim6_adaptSigmaV1_RLsync_100iters_deep2h96_new_covSaving_pop20_parents10"
    printYellow("Run Name: {}".format(run_name))
    # environment:
    env = select_env(args.env_name, use_state_history=args.use_state_history)
    parameters.state_dim = env.observation_space.shape[0]
    parameters.action_dim = env.action_space.shape[0]

    # Shared Memory Buffers:
    g_memory = ReplayMemory(args.mem_size)  # good or elite buffer:
    b_memory = ReplayMemory(args.mem_size)  # bad states:
    n_memory = ReplayMemory(args.mem_size)  # noisy states:

    params_dict = parameters.__dict__
    parameters.stdout()
    if args.should_log:
        import wandb
        print("WandB logging Started")
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,  #"alexanicetzongo",
            dir="./logs",
            name=run_name,
            config=params_dict,
        )
        parameters.save_foldername = str(run.dir)
        print('Saved to:', parameters.save_foldername)
        wandb.config.update({
            "save_foldername": parameters.save_foldername,
            "run_name": run.name,
        }, allow_val_change=True)

    # set the seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # random.seed(args.seed)

    if args.use_state_history:
        printPurple("Using State (Obs) History t-2, t-1 and t")
    # *** TD3 Agent ***#
    if args.use_multiAgent:
        actor = MultiAgentActor(parameters)
        RL_agent = TD3ES(parameters)
    elif args.use_sac:
        actor = SACGaussianActor(parameters)
        RL_agent = SAC(parameters)
    else:
        actor = CONV_ACTOR(parameters) if parameters.use_state_history else Actor(
            parameters, init=True)
        RL_agent = TD3ES(parameters)
    num_params = actor.count_parameters()
    printPurple(">> Actor Architecture: {}".format(actor))
    # rl_actor = Actor(parameters, init=True)
    # actor_t = Actor(parameters, init=True)
    # actor_t.load_state_dict(actor.state_dict())

    td3_num_params = RL_agent.actor.count_parameters()

    # action noise:
    if args.ou_noise:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim=parameters.action_dim,
            mu=args.ou_mu,
            theta=args.ou_theta,
            sigma=args.ou_sigma
        )
    else:
        print("Using Gaussian Noise")
        a_noise = GaussianNoise(
            action_dim=parameters.action_dim,
            sigma=args.gauss_sigma
        )

    # **** ES agent ****#
    printYellow("Using CEM")
    es = CEM(
        num_params=num_params,
        params=parameters,
        mu_init=deepcopy(RL_agent.actor.extract_parameters().cpu().numpy()),
        sigma_init=args.sigma_init,
        pop_size=args.pop_size,
        antithetic=not args.pop_size % 2,
        parents=args.parents,
        elitism=args.elitism,
        adaptation=args.cem_with_adapt,
    )

    print(
        f" Running the environment: {parameters.env_name}\n State_dim: {parameters.state_dim}\n Action_dim: {parameters.action_dim}\n")
    printPurple(
        "Number of parameters or search space: {}".format(num_params))
    printPurple(
        "Number of parameters or search space: {}".format(td3_num_params))
    assert (num_params == td3_num_params)
    # training:
    max_gen_per_iter = args.n_generations
    max_iter = 10
    start_time = time.time()
    gen_cpt = 0
    actor_steps = 0
    rl_actor_steps = 0
    rl_iter = 1
    total_steps = 0
    generation = 1
    r_threshold = -np.inf
    has_new_rl_agent = 0
    # print("num steps", parameters.max_steps)
    print("current step", total_steps)
    # comb_memory = ReplayMemory(args.mem_size)
    # while generation < args.n_generations:
    for iter in range(1, args.max_iter+1):
        # choose a random environment:
        # if iter % 2 == 0:
        #     env = select_env(ENVS[EnvIDs[random.randrange(1, 13)]])
        # else:
        #     env = select_env(ENVS['NOMINAL'])
        printYellow(f"Current Step: {total_steps}")
        printYellow(f"Current Threshold: {r_threshold}")

        # ***** RL actors and critic *******#

        rl_fit, rl_sm_avg, rl_ep_length_avg = -np.inf, -np.inf, -np.inf
        actor_steps = 0
        # *** collect experiences:

        episode = evaluate(
            actor=RL_agent.actor,
            env=env,
            params=parameters,
            g_buffer=g_memory,
            b_buffer=b_memory,
            n_buffer=n_memory,
            # noise=a_noise,
            threshold=r_threshold,
        )
        actor_steps += episode.n_steps
        rl_actor_steps += episode.n_steps
        print("RL Actor steps: {}".format(rl_actor_steps))
        r_threshold = episode.threshold
        # **** train the rl agent:

        rl_iter, td_loss = train_rl(
            agent=RL_agent,
            params=parameters,
            g_buffer=g_memory,
            b_buffer=b_memory,
            n_buffer=n_memory,
            rl_iter=rl_iter,
            rl_transitions=rl_actor_steps,
        )
        rl_eval = dict()
        for env_name in ENVS.values():
            env_test = select_env(
                env_name, use_state_history=args.use_state_history)
            printYellow(f"Testing on {env_name}")
            rl_fit, rl_sm_avg, rl_sm_sd, rl_ep_length_avg, rl_ep_length_sd, rl_steps, rl_episode, threshold = fitness_function(
                actor=RL_agent.actor,
                env=env_test,
                params=parameters,
                # g_buffer=g_memory,
                # b_buffer=b_memory,
                # n_buffer=n_memory,
                n_evals=args.n_evals,
                # noise=a_noise,
                threshold=r_threshold,
            )
            rl_eval[env_name] = {
                'rl_fit': rl_fit,
                'rl_sm_avg': rl_sm_avg,
                'rl_sm_sd': rl_sm_sd,
                'rl_ep_length_avg': rl_ep_length_avg,
                'rl_ep_length_sd': rl_ep_length_sd,
                'rl_steps': rl_steps,
                'rl_episode': rl_episode,
                'threshold': threshold,
            }
        avg_rl_fit_on_all_scene = np.mean(
            [rl_eval[env]['rl_fit'] for env in rl_eval.keys()])
        actor_steps += rl_eval[ENVS['NOMINAL']]['rl_steps']
        r_threshold = rl_eval[ENVS['NOMINAL']]['threshold']
        rl_fit = rl_eval[ENVS['NOMINAL']]['rl_fit']
        printPurple('RL Actor fitness: {}'.format(
            rl_fit))
        printPurple('RL Actor Avg fitness all Scenarios: {}'.format(
            avg_rl_fit_on_all_scene))

        if generation % 4 == 1:
            print(">>> Generation by 4: ", generation % 4)
        if (generation % 4 and generation > 1):
            printPurple('Current Gen: {}'.format(generation))
            if rl_fit > 1.005 * test_mu_score:
                printRed("RL agent inserted to pop")
                has_new_rl_agent += 1
                es.rl_agent = deepcopy(
                    RL_agent.actor.extract_parameters().cpu().numpy())
                es.rl_agent_score = rl_fit
                if rl_fit > test_mu_score:
                    printRed("RL agent surpasses ES agent")
                    es.mu = deepcopy(
                        RL_agent.actor.extract_parameters().cpu().numpy())
            else:
                printRed("ES Mu agent surpasses RL agent")
                if args.elitism:
                    # replace the RL actor with the best_elite so far
                    RL_agent.actor.inject_parameters(es.best_mu_so_far)
                else:
                    RL_agent.actor.inject_parameters(es.mu)

                RL_agent.actor_optim = Adam(
                    RL_agent.actor.parameters(), lr=args.actor_lr)

        printYellow("ES section")

        while generation <= args.n_generations:
            printYellow(f"Generation: {generation} - Iteration: {iter}")
            fit_lst, smAvg_lst, smSd_lst, epAvg_lst, epSd_lst, fit_test = [], [], [], [], [], []
            es_sol = es.ask(parameters.pop_size)
            for candidate in es_sol:
                actor.inject_parameters(candidate)
                fit, sm_avg, sm_sd, ep_length_avg, ep_length_sd, steps, episode, threshold = fitness_function(
                    actor=actor,
                    env=env,
                    params=parameters,
                    g_buffer=g_memory,
                    b_buffer=b_memory,
                    n_buffer=n_memory,
                    n_evals=args.n_evals,
                    # noise=a_noise if generation % 2 == 0 else None,
                    threshold=r_threshold,
                )
                r_threshold = threshold

                actor_steps += steps
                fit_lst.append(fit)
                smAvg_lst.append(sm_avg)
                smSd_lst.append(sm_sd)
                epAvg_lst.append(ep_length_avg)
                epSd_lst.append(ep_length_sd)

                printPurple('Actor fitness: {}'.format(fit))

            es.tell(es_sol, fit_lst)

            # update step counts:
            total_steps += actor_steps
            gen_cpt += generation
            generation += 1

            # stats:
            rl_score = rl_fit if rl_fit else -np.inf
            best_train_fitness = np.array(fit_lst).max()
            worst_train_fitness = np.array(fit_lst).min()
            pop_avg = np.array(fit_lst).mean()
            sm_avg = np.array(smAvg_lst).mean()
            sm_sd = np.array(smSd_lst).std()
            ep_avg = np.array(epAvg_lst).mean()
            ep_sd = np.array(epSd_lst).mean()

            # test == evaluation with elite #TODO: elite or mu
            if args.elitism:
                actor.inject_parameters(es.elite)
                test_elite_score = es.elite_score
                best_elite_score = es.best_elite_so_far_score

            actor.inject_parameters(es.mu)
            test_mu_scores = dict()
            for envi in ENVS.values():
                env_test = select_env(
                    envi, use_state_history=args.use_state_history)

                test_mu_score, _, _, _, _, _, _, _ = fitness_function(
                    actor=actor,
                    env=env_test,
                    params=parameters,
                    g_buffer=g_memory,
                    b_buffer=b_memory,
                    n_buffer=n_memory,
                    n_evals=args.n_evals,
                    # noise=a_noise,
                    threshold=r_threshold,
                )
                test_mu_scores[envi] = test_mu_score

            test_mu_score_all_scene = np.mean(list(test_mu_scores.values()))
            printYellow("Mean validation Score on all scenario: {}".format(
                        test_mu_score_all_scene))
            test_mu_score = test_mu_scores[ENVS['NOMINAL']]
            es.mu_score = test_mu_score
            if test_mu_score > es.best_mu_so_far_score:
                es.best_mu_so_far_score = test_mu_score
                es.best_mu_so_far = deepcopy(es.mu)
                es.best_cov = deepcopy(es.cov)

            printRed("Threshold fit: {}".format(r_threshold))
            printRed("Actor Elite Test Average Fitness: {}".format(
                test_elite_score))
            printRed("Actor Best Elite So Far Fitness: {}".format(
                best_elite_score))
            printRed("Actor Mu Test Average Fitness: {}".format(test_mu_score))
            printRed("Actor Best Mu So Far Fitness: {}".format(
                es.best_mu_so_far_score))

            stats = {
                'avg_mu_fit_on_all_scenario': test_mu_score_all_scene,
                'avg_rl_fit_on_all_scenario': avg_rl_fit_on_all_scene,
                'total_steps': total_steps,
                'rl_avg_score': rl_score,
                'rl_td_loss': td_loss,
                'rl_sm': rl_eval[ENVS['NOMINAL']]['rl_sm_avg'],
                'rl_ep_length': rl_eval[ENVS['NOMINAL']]['rl_ep_length_avg'],
                'ea_avg_score': np.mean(fit_lst),
                'best_train_fitness': best_train_fitness,
                'test_elite_score': test_elite_score,
                'test_best_elite_so_far_score': best_elite_score,
                'test_mu_score': test_mu_score,
                'test_best_mu_so_far_score': es.best_mu_so_far_score,
                'pop_min': worst_train_fitness,
                'pop_avg': pop_avg,
                'avg_smoothness': sm_avg,
                'smoothness_sd': sm_sd,
                'avg_ep_length': ep_avg,
                'ep_length_sd': ep_sd,
                'sigma_value': es.sigma,
            }

            stats['time'] = time.time() - start_time

            # log the stats to WandB
            if args.should_log:
                wandb.log(stats)

        print(stats)

        print("Total steps: {}".format(total_steps))
        args.n_generations += max_gen_per_iter

    # save the mu agent:
    if args.save_models:
        if args.elitism:
            # ***** saving best cov
            log_dir = os.path.join(parameters.save_foldername)
            save_path = log_dir / Path('best_cov.csv')
            print(save_path)
            with open(save_path, 'w+', encoding='utf-8') as fp:
                np.savetxt(fp, es.cov)

            print(f'>> Saved best cov in: {save_path} \n')
            fp.close()

            # filename = 'best_cov_' + str(total_steps) + '.csv'
            # np.savetxt(os.path.join(parameters.save_foldername), es.best_cov)
            # print('>> Saved best cov in ' + str(filename) + '\n')

            # ***** Saving Elite agent
            actor.inject_parameters(es.elite)
            torch.save(actor.state_dict(), os.path.join(
                parameters.save_foldername, 'elite_agent.pkl'
            ))
            history = fitness_function(
                actor=actor,
                env=env,
                params=parameters,

                n_evals=args.n_evals,
                # noise=a_noise,
            )[-2].get_history()
            filename = 'elite_state_history_' + str(total_steps) + '.txt'
            np.savetxt(os.path.join(parameters.save_foldername,
                                    filename), history, header=str(total_steps))
            print('>> Saved Elite state history in ' + str(filename) + '\n')

            # ***** Saving best elite seen
            actor.inject_parameters(es.best_elite_so_far)
            torch.save(actor.state_dict(), os.path.join(
                parameters.save_foldername, 'best_elite_agent.pkl'
            ))
            history = fitness_function(
                actor=actor,
                env=env,
                params=parameters,

                n_evals=args.n_evals,
                # noise=a_noise,
            )[-2].get_history()
            filename = 'best_elite_state_history_' + \
                str(total_steps) + '.txt'
            np.savetxt(os.path.join(parameters.save_foldername,
                                    filename), history, header=str(total_steps))
            print('>> Saved Best Elite state history in ' +
                  str(filename) + '\n')

        # ******** Saving Mu agent
        actor.inject_parameters(es.mu)
        history = fitness_function(
            actor=actor,
            env=env,
            params=parameters,
            n_evals=args.n_evals,
            # noise=a_noise,
        )[-2].get_history()
        torch.save(actor.state_dict(), os.path.join(
            parameters.save_foldername, 'mu_agent.pkl'
        ))
        filename = 'mu_state_history_' + str(total_steps) + '.txt'
        np.savetxt(os.path.join(parameters.save_foldername,
                                filename), history, header=str(total_steps))
        print('>> Saved Mu state history in ' + str(filename) + '\n')

        # ******** saving best MU seen
        actor.inject_parameters(es.best_mu_so_far)
        history = fitness_function(
            actor=actor,
            env=env,
            params=parameters,
            n_evals=args.n_evals,
            # noise=a_noise,
        )[-2].get_history()
        torch.save(actor.state_dict(), os.path.join(
            parameters.save_foldername, 'best_mu_agent.pkl'
        ))
        filename = 'mu_state_history_' + str(total_steps) + '.txt'
        np.savetxt(os.path.join(parameters.save_foldername,
                                filename), history, header=str(total_steps))
        print('>> Saved Best Mu state history in ' + str(filename) + '\n')

        # ****** Saving RL agent ****** #
        if has_new_rl_agent:
            actor.inject_parameters(es.rl_agent)
        history = fitness_function(
            actor=actor if has_new_rl_agent else RL_agent.actor,
            env=env,
            params=parameters,
            n_evals=args.n_evals,
            # noise=a_noise,
        )[-2].get_history()
        torch.save(RL_agent.actor.state_dict(), os.path.join(
            parameters.save_foldername, 'rl_agent.pkl'
        ))
        filename = 'rl_state_history_' + str(total_steps) + '.txt'
        np.savetxt(os.path.join(parameters.save_foldername,
                                filename), history, header=str(total_steps))
        print('>> Saved Rl state history in ' + str(filename) + '\n')
