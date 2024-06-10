from torch.distributions import Normal
import torch
import torch.nn as nn
from torch.optim import Adam
from core_algorithms.model_utils import activations, LayerNorm, hard_update, soft_update, is_lnorm_key
from parameters_es import ESParameters
from core_algorithms.replay_memory import ReplayMemory
from torch.nn import functional as F


class BaseNN(nn.Module):
    def __init__(self, args: ESParameters) -> None:
        super(BaseNN, self).__init__()

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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SACGaussianActor(BaseNN):
    def __init__(self, args: ESParameters):
        super(SACGaussianActor, self).__init__(args)
        self.h = args.actor_hidden_size
        self.L = args.actor_num_layers
        self.a = activations[args.activation_actor.lower()]
        self.args = args
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.h,
                              kernel_size=(args.state_length, 1))
        self.fc = nn.Linear(self.h*args.state_dim, self.h)

        layers = []
        for _ in range(self.L):
            layers.extend([
                nn.Linear(self.h, self.h),
                LayerNorm(self.h),
                self.a
            ])

        self.net = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(self.h, args.action_dim)
        self.log_std_linear = nn.Linear(self.h, args.action_dim)
        self.apply(init_weights)
        self.to(args.device)

    def forward(self, state: torch.tensor):
        B = state.shape[0] if state.ndim > 2 else 1
        # state = torch.FloatTensor(state)
        x = self.conv(state.view(B, 1, self.args.state_length, -1))
        x = self.a(self.fc(x.view(B, -1)))
        x = self.net(x)
        mean = self.a(self.mean_linear(x))
        log_std = self.a(self.log_std_linear(x))
        log_std = torch.clamp(
            log_std, min=self.args.log_std_min, max=self.args.log_std_max)
        return mean, log_std

    def sample(self, state: torch.tensor):
        if state.ndim > 2:
            B, L, _ = state.shape
            state = torch.FloatTensor(state.reshape(B, L, -1))
        elif state.ndim == 2:
            L, _ = state.shape
            state = torch.FloatTensor(state.reshape(1, L, -1))
        else:
            state = torch.FloatTensor(state.reshape(1, -1))
        state = state.to(self.args.device)
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        return action, log_prob, mean

    def select_action(self, state: torch.tensor):

        action, _, _ = self.sample(state)
        return action.cpu().data.numpy().flatten()


class SACDeterministicActor(BaseNN):
    def __init__(self, args: ESParameters) -> None:
        super(SACDeterministicActor, self).__init__(args)
        self.h = args.actor_hidden_size
        self.L = args.actor_num_layers
        self.a = activations[args.activation_actor.lower()]
        self.args = args
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.h,
                              kernel_size=(args.state_length, 1))
        self.fc = nn.Linear(self.h*args.state_dim, self.h)

        layers = []
        for _ in range(self.L):
            layers.extend([
                nn.Linear(self.h, self.h),
                LayerNorm(self.h),
                self.a
            ])

        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(self.h, args.action_dim)
        self.noise = torch.Tensor(args.action_dim)
        self.apply(init_weights)
        self.to(args.device)

    def forward(self, state: torch.tensor):
        B = state.shape[0] if state.ndim > 2 else 1
        x = state.view(B, 1, self.args.state_length, -1)
        x = self.conv(x)
        x = self.a(self.fc(x.view(B, -1)))
        x = self.net(x)
        mean = self.a(self.mean(x))
        return mean

    def sample(self, state: torch.tensor):

        if state.ndim > 2:
            B, L, _ = state.shape
            state = torch.FloatTensor(state.reshape(B, L, -1))
        elif state.ndim == 2:
            L, _ = state.shape
            state = torch.FloatTensor(state.reshape(1, L, -1))
        else:
            state = torch.FloatTensor(state.reshape(1, -1))
        state = state.to(self.args.device)
        action = self.forward(state)
        noise = self.noise.normal_(0., self.args.noise_sd)
        noise = noise.clamp(-self.args.noise_clip, self.args.noise_clip)
        action += noise
        return action, torch.tensor(0.), action-noise

    def select_action(self, state: torch.tensor):
        action, _, _ = self.sample(state)
        return action.cpu().data.numpy().flatten()


class SACQNetwork(BaseNN):
    def __init__(self, args: ESParameters) -> None:
        super(SACQNetwork, self).__init__(args)
        self.h = args.critic_hidden_size[0]
        self.state_length = args.state_length
        self.input_dim = args.state_dim
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=self.h, kernel_size=(self.state_length, 1))
        self.fc = nn.Linear(self.h*self.input_dim, self.h)
        self.a = activations[args.activation_critic.lower()]

        # critic 1
        self.bn1 = nn.BatchNorm1d(self.h + args.action_dim)
        self.linear11 = nn.Linear(self.h + args.action_dim, self.h)
        self.linear1 = nn.Linear(self.h, 1)
        self.ln1 = LayerNorm(self.h)

        # critic 2
        self.bn2 = nn.BatchNorm1d(self.h + args.action_dim)
        self.linear22 = nn.Linear(self.h + args.action_dim, self.h)
        self.linear2 = nn.Linear(self.h, 1)
        self.ln2 = LayerNorm(self.h)

        self.apply(init_weights)
        self.to(args.device)

    def forward(self, state, action):
        B = state.shape[0] if state.ndim > 2 else 1
        x = state.view(B, 1, self.state_length, -1)
        x = self.conv(x)
        x = self.a(self.fc(x.view(B, -1)))
        x = torch.cat([x, action], dim=1)

        x1 = self.a(self.linear11(self.bn1(x)))
        x1 = self.ln1(x1)
        x1 = self.linear1(x1)

        x2 = self.a(self.linear22(self.bn2(x)))
        x2 = self.ln2(x2)
        x2 = self.linear2(x2)

        return x1, x2


class SAC(object):
    def __init__(self, args: ESParameters):
        self.args = args

        self.critic = SACQNetwork(args).to(args.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)
        self.critic_target = SACQNetwork(args).to(args.device)
        hard_update(self.critic_target, self.critic)
        if args.policy_type == "Gaussian":
            if args.automatic_entropy_tuning:
                self.target_entropy = - \
                    torch.Tensor(args.action_dim).to(args.device)
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=args.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.critic_lr)

            self.actor = SACGaussianActor(args).to(args.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = SACDeterministicActor(args).to(args.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

    def update_parameters(self, batch, iteration: int):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch
        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        done_batch = done_batch.to(self.args.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_action)
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - self.alpha * next_log_pi
            next_q_value = reward_batch + \
                (1 - done_batch) * self.args.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        if self.args.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi +
                           self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        else:
            alpha_loss = torch.tensor(0.).to(self.args.device)

        if iteration % self.args.policy_update_freq == 0:
            soft_update(self.critic_target, self.critic, self.args.tau)

        return policy_loss.data.cpu().numpy(), qf_loss.data.cpu().numpy(), alpha_loss.data.cpu().numpy()
