import argparse
import os
import random
import time
from distutils.util import strtobool
from core_algorithms.utils import calc_smoothness
from environments.aircraftenv import printPurple, printYellow
from environments.config import select_env
import gymnasium as gym
import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument('--add-sm-to-reward', action='store_true', default=False, help='to add smoothness metric to the reward function')
    parser.add_argument('--use-scaled-obs', action='store_true', default=False, help='scale the observed states')
    parser.add_argument('--multiple-envs', action='store_true', default=False, help='train the agent on multiple environments simultaneously')
    parser.add_argument('--without-beta', action='store_true', default=False, help='train the agent without the side-slip angle in the observed state.')

    args = parser.parse_args()
    if args.multiple_envs:
        args.num_envs = 13
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

    return args


def make_env(gym_id, seed, idx, capture_video, run_name, eval=False, add_sm=False, use_scaled_obs=False, without_beta=False):
    def thunk():
        if gym_id.startswith("PHlab"):
            env = select_env(
                environment_name=gym_id,
                conform_with_sb=True,
                add_sm_to_reward=add_sm,
                use_scaled_obs=use_scaled_obs,
                without_beta=without_beta
            )
            if eval:
                env.set_eval_mode(t_max=80)
        else:
            env = gym.make(gym_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(
        #     env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(
        #     env, lambda reward: np.clip(reward, -10, 10))
        # env.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """DNN agent with the actor and critic neural network configuration"""
    def __init__(self, envs, eval: bool = False):
        super(Agent, self).__init__()
        self.eval = eval
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(
                envs.single_action_space.shape)), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(
            1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def forward(self, x):
        action_mean = self.actor_mean(x)
        value = self.critic(x)
        # log_std = torch.exp(self.actor_logstd.expand_as(
        #     action_mean.reshape(1, -1)))
        log_prob = torch.zeros_like(action_mean, requires_grad=True).sum(-1)
        prob_entropy = torch.zeros_like(
            action_mean, requires_grad=True).sum(-1)
        return action_mean, log_prob, prob_entropy, value, action_mean

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)

        action_logstd = self.actor_logstd.expand_as(action_mean.reshape(
            1, -1)) if self.eval else self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x), action_mean

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_parameters(self):
        # return [p for p in self.parameters() if p.requires_grad]
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32)
        count = 0
        for name, param in self.named_parameters():
            if len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count+sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    def inject_parameters(self, params):
        # for p, new_p in zip(self.parameters(), params):
        #     if p.requires_grad:
        #         p.data = new_p.data
        count = 0
        pvec = torch.tensor(params)
        for name, param in self.named_parameters():
            # only alter W -- skip norms and biases:
            if len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count+sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz


if __name__ == "__main__":
    args = parse_args()
    if args.without_beta:
        printPurple("States Without Side-slip")

    if args.use_scaled_obs:
        printYellow("Using scaled observations")
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}_addsmtorw_maxdiffsm_{args.add_sm_to_reward}__multipleEnvs_{args.multiple_envs}_singleENV__totalTimesteps_{args.total_timesteps}__useScaledObs_{args.use_scaled_obs}__NoBeta"
    if args.track:
        import wandb

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.multiple_envs:
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
        e_inds = np.arange(13)
        # np.random.shuffle(e_inds)
        envs = gym.vector.SyncVectorEnv(
            [make_env(gym_id=envs_name[ind][0], seed=args.seed+int(ind), idx=int(ind), capture_video=args.capture_video, run_name=run_name, eval=False, add_sm=args.add_sm_to_reward, use_scaled_obs=args.use_scaled_obs)
             for ind in e_inds]
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, add_sm=args.add_sm_to_reward, use_scaled_obs=args.use_scaled_obs, without_beta=args.without_beta)
             for i in range(args.num_envs)]
        )
        printPurple(envs.single_observation_space.shape)

    assert isinstance(envs.single_action_space,
                      gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        actions_lst = []  # TODO: added for smoothness eval
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, action_mean = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # TODO: added for smoothness eval
            actions_lst.append(action.detach().cpu().numpy())

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncated, info = envs.step(
                action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(done).to(device)

            # for item in info:

            if "final_info" in info.keys():
                # print(info["final_info"])
                for item in info["final_info"]:
                    if item is not None:
                        if "episode" in item.keys():
                            # print(info["final_info"])
                            smoothness = np.mean(calc_smoothness(
                                np.asarray(actions_lst)))
                            print(
                                f"global_step={global_step}, episodic_return={item['episode']['r']}")
                            print(
                                f"Episodic smoothness sampled actions ={smoothness}")
                            writer.add_scalar("charts/episodic_return",
                                              item['episode']['r'], global_step)
                            writer.add_scalar("charts/smoothness",
                                              smoothness, global_step)
                            writer.add_scalar("charts/episodic_length",
                                              item["episode"]['l'], global_step)
                            break
            # if "episode" in info.keys():
            #     print(
            #         f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return",
            #                       info["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length",
            #                       info["episode"]["l"], global_step)
            #     break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
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
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    torch.save(agent.state_dict(), f"agents/{run_name}.pkl")
