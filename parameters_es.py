from pprint import pprint
import os
import torch

import numpy as np


class ESParameters:
    def __init__(self, conf={}, init=True):
        if not init:
            return

        # setting the device:
        self.device = torch.device("cpu") if hasattr(
            conf, 'disable_cuda') and conf.disable_cuda else torch.device("cuda")
        print('Current device: %s' % self.device)

        self.env_name = conf.env_name if hasattr(
            conf, 'env') else 'PHlab_attitude_nominal'
        self.use_state_history = conf.use_state_history if hasattr(
            conf, 'use-state-history') else False
        self.continue_training = conf.continue_training if hasattr(
            conf, 'continue-training') else False
        self.fixed_softmax_cem = conf.fixed_softmax_cem if hasattr(
            conf, 'fixed-softmax-cem') else False
        self.state_length = conf.state_length if hasattr(
            conf, 'state-length') else 1
        self.should_log = conf.should_log if hasattr(
            conf, 'should-log') else False
        self.run_name = conf.run_name if hasattr(
            conf, 'run-name') else 'default'

        # =========== TD3 Params ===========
        self.use_td3 = conf.use_td3 if hasattr(conf, 'use-td3') else True
        self.use_multiAgent = conf.use_multiAgent if hasattr(
            conf, 'use-multiAgent') else False
        self.use_rnn = conf.use_rnn if hasattr(conf, "use-rnn") else False
        self.use_lstm = conf.use_lstm if hasattr(conf, "use-lstm") else False
        self.policy_noise = conf.policy_noise if hasattr(
            conf, 'policy-noise') else 0.2
        self.noise_clip = conf.noise_clip if hasattr(
            conf, 'noise-clip') else 0.5
        self.policy_update_freq = conf.policy_update_freq if hasattr(
            conf, 'policy-update-freq') else 3

        # =========== PPO Params ===========
        self.total_timesteps = conf.total_timesteps if hasattr(
            conf, 'total-timesteps') else 1_000_000
        self.timesteps_per_batch = conf.timesteps_per_batch if hasattr(
            conf, 'timesteps-per-batch') else 10000
        self.max_timesteps_per_episode = conf.max_timesteps_per_episode if hasattr(
            conf, 'max-timesteps-per-episode') else 2000
        self.n_updates_per_iteration = conf.n_updates_per_iteration if hasattr(
            conf, 'n-updates-per-iteration') else 5
        self.lam = conf.lam if hasattr(conf, 'lam') else 0.97
        self.init_action_std = conf.init_action_std if hasattr(
            conf, 'init-action-std') else 0.7071
        self.action_std_decay_rate = conf.action_std_decay_rate if hasattr(
            conf, 'action-std-decay-rate') else 0.95
        self.action_std_min = conf.action_std_min if hasattr(
            conf, 'action-std-min') else 0.01
        self.entropy_coef = conf.entropy_coef if hasattr(
            conf, 'entropy-coef') else 0.0001
        self.target_kl = conf.target_kl if hasattr(conf, 'target-kl') else 0.02

        self.num_minibatches = conf.num_minibatches if hasattr(
            conf, 'num-minibatches') else 512
        self.max_grad_norm = conf.max_grad_norm if hasattr(
            conf, 'max-grad-norm') else 0.5
        self.eps_clip = conf.eps_clip if hasattr(conf, 'eps-clip') else 0.2

        # =========== SAC Params ===========
        self.log_std_min = -.25
        self.log_std_max = 0.25
        self.epsilon = 1e-6
        self.policy_type = "Gaussian"  # or deterministic
        self.automatic_entropy_tuning = True
        # ============ Gaussian Noise ==========
        self.gauss_sigma = conf.gauss_sigma if hasattr(
            conf, 'gauss-sigma') else 0.1

        # ============= OU Noise ==============
        self.ou_noise = conf.ou_noise if hasattr(conf, 'ou-noise') else False
        self.ou_theta = conf.ou_theta if hasattr(conf, 'ou-theta') else 0.15
        self.ou_sigma = conf.ou_sigma if hasattr(conf, 'ou-sigma') else 0.2
        self.ou_mu = conf.ou_mu if hasattr(conf, 'ou-mu') else 0.0

        # ============ ES Params ===========
        self.use_cem = conf.use_cem if hasattr(conf, 'use-cem') else True
        self.cem_with_adapt = conf.cem_with_adapt if hasattr(
            conf, 'cem-with-adapt') else True
        self.pop_size = conf.pop_size if hasattr(conf, 'pop-size') else 10
        self.parents = conf.parents if hasattr(
            conf, 'parents') else self.pop_size//2
        self.elitism = conf.elitism if hasattr(conf, 'elitism') else True
        self.n_grad = conf.n_grad if hasattr(conf, 'n-grad') else 0
        self.n_noisy = conf.n_noisy if hasattr(conf, 'n-noisy') else 0
        self.sigma_init = conf.sigma_init if hasattr(
            conf, 'sigma-init') else 0.3
        self.sigma_decay = conf.sigma_decay if hasattr(
            conf, 'sigma-decay') else 0.999  # default=0.999
        self.sigma_limit = conf.sigma_limit if hasattr(
            conf, 'sigma-limit') else 0.001
        self.damp = conf.damp if hasattr(conf, 'damp') else 1e-3
        self.damp_limit = conf.damp_limit if hasattr(
            conf, 'damp-limit') else 1e-5
        self.mult_noise = conf.mult_noise if hasattr(
            conf, 'mult-noise') else False

        # CMAES:
        self.weight_decay = conf.weight_decay if hasattr(
            conf, 'weightdecay') else 0.01  # for CMAES
        # self.sigma_init = conf.sigma_init if hasattr(
        #     conf, "sigma_init") else 0.3

        # ============= Training Params =================
        # Number of experiences to use for each training step:
        self.batch_size = conf.batch_size if hasattr(
            conf, 'batch-size') else 100  # 64 for TD3 alone
        self.n_evals = conf.n_evals if hasattr(conf, 'n-evals') else 2
        self.n_generations = conf.n_generations if hasattr(
            conf, 'ngenerations') else 100
        # self.max_steps = conf.max_steps if hasattr(
        #     conf, 'max_steps') else 100000  # num of steps to run:
        # frames accumulated before grad updates:
        self.start_steps = conf.start_steps if hasattr(
            conf, 'start-steps') else 10_000
        self.max_iter = conf.max_iter if hasattr(conf, 'max-iter') else 10
        self.sample_ratio = conf.sample_ratio if hasattr(
            conf, 'sample-ratio') else [0.8, 0.1, 0.1]
        self.g_batch_size = int(np.ceil(self.sample_ratio[0]*self.batch_size))
        self.b_batch_size = int(np.ceil(self.sample_ratio[1]*self.batch_size))
        self.n_batch_size = int(np.ceil(self.sample_ratio[2]*self.batch_size))
        # buffer size:
        self.mem_size = conf.mem_size if hasattr(
            conf, 'mem-size') else 1_000_000
        # number of noisy evaluations:

        # ================ FDI =================
        self.fdi_hidden_size = conf.fdi_hidden_size if hasattr(
            conf, "fdi-hidden-size") else 64
        self.fdi_num_layers = conf.fdi_num_layers if hasattr(
            conf, "fdi-num-layers") else 2
        self.fdi_out_size = conf.fdi_out_size if hasattr(
            conf, "fdi-out-size") else 6
        self.fdi_pop_size = conf.fdi_pop_size if hasattr(
            conf, "fdi-pop-size") else 5
        self.fdi_parents = conf.fdi_parents if hasattr(
            conf, "fdi-parents") else 3

        # ============= misc ====================

        self.seed = conf.seed if hasattr(conf, "seed") else 7
        # model save frequency:
        self.period = conf.period if hasattr(conf, 'period') else 1000

        self.gamma = 0.999
        self.noise_sd = 0.33

        # soft update:
        self.tau = 0.005
        # hidden layer:
        self.actor_num_layers = conf.actor_num_layers if hasattr(
            conf, 'actor-num-layers') else 2
        # 72 for SERL10 or SERL50, 96 for TD3
        self.actor_hidden_size = conf.actor_hidden_size if hasattr(
            conf, 'actor-hidden-size') else 32
        self.actor_lr = conf.actor_lr if hasattr(
            conf, 'actor-lr') else 0.001  # for TD3 alone 0.0000482

        self.critic_num_layers = conf.critic_num_layers if hasattr(
            conf, 'critic-num-layers') else 2
        self.critic_hidden_size = conf.critic_hidden_size if hasattr(
            conf, 'critic-hidden-size') else [32, 64]  # [200, 300]
        self.critic_lr = conf.critic_lr if hasattr(
            conf, 'critic-lr') else 0.001

        self.activation_actor = 'tanh'
        self.activation_critic = 'tanh'
        self.nonlin_activation = 'relu'  # for SERL10 or SERL50, 'relu' for TD3

        # prioritized experience replay:
        self.per = conf.per if hasattr(conf, "per") else False
        if self.per:
            self.replace_old = True
            self.alpha = 0.7
            self.beta_zero = 0.5

        # CAPS: Condition for Action Policy Smoothness:
        self.use_caps = conf.use_caps if hasattr(conf, "use-caps") else False
        # print("Using CAPS: ", self.use_caps)

        # save the results:
        self.state_dim = None
        self.action_dim = None
        self.action_high = None
        self.action_low = None
        self.save_foldername = './logs'
        self.should_log = conf.should_log if hasattr(
            conf, "should_log") else False

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

    def write_params(self, stdout=False):
        """ Transfer the parameters to a state dictionary
            Args:
                stdout: whether to print the parameters Defaults to True
        """
        if stdout:
            params = pprint.pformat(vars(self), indent=4)
            print(params)
        return self.__dict__

    def update_from_dict(self, new_config_dict: dict):
        """ Update the parameters from a dictionary
            Args:
                new_config_dict: the new configuration dictionary
        """
        self.__dict__.update(new_config_dict)

    def stdout(self) -> None:
        keys = ['save_foldername', 'seed', 'batch_size',
                'actor_lr', 'critic_lr', 'use_state_history', 'state_length', 'total_timesteps', 'timesteps_per_batch', 'n_updates_per_iteration',
                'actor_num_layers', 'actor_hidden_size', 'critic_hidden_size', 'activation_actor',
                'activation_critic', 'pop_size',
                'n_evals', 'n_generations',
                'sigma_init', 'sigma_decay', 'start_steps', 'max_iter', 'sample_ratio'
                ]
        _dict = {}
        for k in keys:
            _dict[k] = self.__dict__[k]

        pprint(_dict)
