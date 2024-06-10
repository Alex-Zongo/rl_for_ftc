import random
import time
import numpy as np
import argparse
from distutils.util import strtobool
import os
import gymnasium as gym
import torch
from core_algorithms.fdi import FDI, make_env
# from FaultDI import make_env, FDI
from core_algorithms.utils import calc_smoothness
# from environments.config import select_env
from ppo_continuous_actions import layer_init, Agent


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="PHlab_attitude_nominal",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ES-PPO for Smooth Fault-Tolerant Control",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='alexanicetzongo',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2001,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10*2,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument('--add-sm-to-reward', action='store_true', default=False)
    parser.add_argument('--multiple-envs', action='store_true', default=False)
    parser.add_argument('--s-k', type=float, default=0.6, help="smoothing factor added to the reward")

    args = parser.parse_args()
    if args.multiple_envs:
        args.num_envs = 13
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

    return args

#*** pre-trained FDI on the fault environment ****
fdi_continuously_trained = {
    0: "fdi_continuous_training__1__1707819572__totalTimesteps_2000000.pkl",
}


class SmoothFilter(torch.nn.Module):

    # 2 layers with output of dimension 3
    # a value function to evaluate the filter:
    # input: c_action, fdi_p
    def __init__(self, envs, args):
        super(SmoothFilter, self).__init__()
        self.args = args
        self.sm_filter = torch.nn.Sequential(
            # input: combine c_action and fdi_p || or observation
            layer_init(torch.nn.Linear(
                np.prod(envs.single_observation_space.shape)+3, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            # ========================
            layer_init(torch.nn.Linear(
                64, envs.single_action_space.shape[0]), std=0.01),
            torch.nn.Tanh(),
        )
        self.sm_filter_logstd = torch.nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.q_filter = torch.nn.Sequential(
            # input: combine c_action and fdi_p || or observation
            layer_init(torch.nn.Linear(
                np.prod(envs.single_observation_space.shape), 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            # ========================
            layer_init(torch.nn.Linear(64, 1), std=1.0),
        )

    def get_q_filter_value(self, obs):
        # x = torch.cat([obs, fdi_p], dim=-1)
        return self.q_filter(obs)

    def get_sm_param_and_value(self, obs, fdi_p, action=None):
        x = torch.cat([obs, fdi_p], dim=-1)
        filter_mean = self.sm_filter(x)
        filter_logstd = self.sm_filter_logstd.expand_as(
            filter_mean) if not self.args.eval else self.sm_filter_logstd.expand_as(filter_mean.reshape(1, -1))
        filter_std = torch.exp(filter_logstd)
        prob = torch.distributions.Normal(filter_mean, filter_std)
        if action is None:
            action = prob.sample()
        return action, prob.log_prob(action).sum(-1), prob.entropy(), self.q_filter(obs), filter_mean, prob.log_prob(filter_mean).sum(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = parse_args()
    args.use_relu = False
    args.eval = False
    args.add_sm_to_reward = True
    # args.deep = True
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}__totalTimesteps_{args.total_timesteps}_Epochs{args.update_epochs}_obsAndfdi_SampledKp_smfftAndRew{args.s_k}__fdiMinusCaction_SingleNormENV"

    if args.track:
        import wandb
        from torch.utils.tensorboard import SummaryWriter

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)
    envs_name = {
        0: ["PHlab_attitude_nominal", torch.Tensor([int(0)]).to(device)],
        1: ["PHlab_attitude_sa", torch.Tensor([int(1)]).to(device)],
        2: ["PHlab_attitude_se", torch.Tensor([int(1)]).to(device)],
        3: ["PHlab_attitude_be", torch.Tensor([int(1)]).to(device)],
        4: ["PHlab_attitude_jr", torch.Tensor([int(1)]).to(device)],
        5: ["PHlab_attitude_cg", torch.Tensor([int(1)]).to(device)],
        6: ["PHlab_attitude_low-q", torch.Tensor([int(1)]).to(device)],
        7: ["PHlab_attitude_high-q", torch.Tensor([int(1)]).to(device)],
        8: ["PHlab_attitude_ice", torch.Tensor([int(1)]).to(device)],
        9: ["PHlab_attitude_noise", torch.Tensor([int(1)]).to(device)],
        10: ["PHlab_attitude_cg-shift", torch.Tensor([int(1)]).to(device)],
        11: ["PHlab_attitude_cg-for", torch.Tensor([int(1)]).to(device)],
        12: ["PHlab_attitude_gust", torch.Tensor([int(1)]).to(device)],
    }

    controllers_names = {
        0: 'PHlab_attitude_nominal__ppo_SingleEnvSync_gpu_sampledAction__7__1703314636.pkl',
        1: 'PHlab_attitude_nominal__ppo_continous_actions__7__1707401136_addsmtorw_maxdiffsm_False__multipleEnvs_True__totalTimesteps_10000000.pkl',
    }

    # env_inds = np.arange(1)
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(gym_id=envs_name[ind][0], seed=args.seed+int(ind), add_sm=args.add_sm_to_reward, eval=False)
    #      for ind in env_inds]
    # )
    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id=envs_name[0][0], seed=args.seed+i, add_sm=args.add_sm_to_reward, eval=False)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space,
                      gym.spaces.Box), "only continuous action space is supported"

    controller = Agent(envs=envs, eval=False).to(device)
    controller.load_state_dict(
        torch.load('agents/'+controllers_names[0])
    )
    # for param in controller.parameters():
    #     param.requires_grad = False

    # **** load a pretrained FDI model ****#
    fdi = FDI(envs, args).to(device)
    fdi.load_state_dict(torch.load('agents/' + fdi_continuously_trained[0]))

    # for param in fdi.parameters():
    #     param.requires_grad = False

    filter = SmoothFilter(envs, args).to(device)
    print(filter)
    print(f"Smooth Filter has {filter.count_parameters()} parameters")

    optimizer = torch.optim.Adam(
        filter.parameters(), lr=args.learning_rate, eps=1e-5)

    # args.num_envs = len(env_inds)
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    c_actions = torch.zeros((args.num_steps, args.num_envs) +
                            envs.single_action_space.shape).to(device)
    fdi_kps = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    sms = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    # s_k = 0.6
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates+1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        actions_lst = []
        actions_mean_lst = []
        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                c_action = controller.get_action_and_value(next_obs)[-1]
                fdi_p = fdi.forward(next_obs, c_action)[-1]
                kp, logp, _, value, kp_mean, logp_mean = filter.get_sm_param_and_value(
                    next_obs, fdi_p)
                action = c_action + kp*(fdi_p-c_action)
                # print(
                #     f"c_action: {c_action}, fdi_p: {fdi_p}, kp: {kp}, action: {action}")
                # if step % 100:
                #     print(f"Diff: {action-c_action}")
                values[step] = value.flatten()
            c_actions[step] = c_action
            fdi_kps[step] = fdi_p
            actions[step] = action
            logprobs[step] = logp_mean

            actions_lst.append(action.detach().cpu().numpy())
            actions_mean_lst.append(
                (c_action+kp_mean*(fdi_p-c_action)).detach().cpu().numpy())
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards[step] = (1.0-args.s_k)*torch.tensor(reward).to(device).view(-1) + \
                args.s_k * torch.tensor(info['sm_fft']).to(device).view(-1)
            sms[step] = torch.tensor(info['sm_fft']).to(device).view(-1)

            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)
            if "final_info" in info.keys():
                # print(info["final_info"])
                for item in info["final_info"]:
                    if item is not None:
                        if "episode" in item.keys():
                            # print(info["final_info"])
                            smoothness = np.mean(calc_smoothness(
                                np.asarray(actions_lst)))
                            smoothness_mean = np.mean(calc_smoothness(
                                np.asarray(actions_mean_lst)))

                            print(
                                f"global_step={global_step}, episodic_return={item['episode']['r']} | {rewards.sum(0)} | {sms.sum(0)} |")
                            print(
                                f"Episodic smoothness sampled actions ={smoothness} | mean actions = {smoothness_mean}")

                            if args.track:
                                writer.add_scalar("charts/episodic_return",
                                                  item['episode']['r'], global_step)
                                writer.add_scalar("charts/smoothness",
                                                  smoothness, global_step)
                                writer.add_scalar("charts/episodic_length",
                                                  item["episode"]['l'], global_step)
                            break

        with torch.no_grad():
            last_c_action = controller.get_action_and_value(next_obs)[-1]
            last_fault_kp = fdi.forward(next_obs, last_c_action)[-1]
            next_value = filter.get_q_filter_value(
                next_obs).reshape(1, -1)

            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * \
                        nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * \
                        args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * \
                        nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_c_actions = c_actions.reshape((-1,) + envs.single_action_space.shape)
        b_fdi_kps = fdi_kps.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _, newlogprob_mean = filter.get_sm_param_and_value(
                    b_obs[mb_inds], b_fdi_kps[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y
        if args.track:
            writer.add_scalar("charts/learning_rate",
                              optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss",
                              pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy",
                              entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl",
                              old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl",
                              approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac",
                              np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance",
                              explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step /
                                                (time.time() - start_time)), global_step)

    envs.close()
    if args.track:
        writer.close()
        torch.save(filter.state_dict(), f"agents/{run_name}.pkl")
