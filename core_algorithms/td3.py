import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
# from torch.optim.lr_scheduler import ExponentialLR

from core_algorithms.model_utils import activations, LayerNorm, hard_update, soft_update, is_lnorm_key
from core_algorithms.some_actor_model import CONV_ACTOR, CONV_CRITIC
from core_algorithms.multi_agent import MultiAgentActor
# from parameters import Parameters
from parameters_es import ESParameters
# from core_algorithms.replay_memory import ReplayMemory
# from core_algorithms.genetic_agent import Actor
# import logging

MAX_GRAD_NORM = 10


class Actor(nn.Module):
    def __init__(self, args: ESParameters, init=False):
        super(Actor, self).__init__()
        self.args = args
        h = self.args.actor_hidden_size
        self.L = self.args.actor_num_layers
        self.activation = activations[args.activation_actor.lower()]

        layers = []
        layers.extend([
            nn.Linear(args.state_dim, h),
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
            nn.Linear(h, args.action_dim),
            nn.Tanh()
        ])
        # # print(*layers)
        self.net = nn.Sequential(*layers)
        self.to(args.device)

    def forward(self, state: torch.tensor):

        return self.net(state)

    def select_action(self, state: torch.tensor):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        """ How different is the new action compared to the last one """
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(
            torch.sum((action_batch - self.forward(state_batch))**2, dim=1))
        self.novelty = novelty.item()
        return self.novelty

    def extract_grad(self):
        """ Current pytorch gradient in same order as genome's flattened parameter vector """
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count+sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

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


class Critic(nn.Module):
    def __init__(self, args: ESParameters):
        super(Critic, self).__init__()
        self.args = args
        # layer sizes:
        # Original PDERL values (no tuning done): l1=200; l2=300; l3=l2
        # l1, l2 = 64, 64  # only worked for TD3-only control:
        [l1, l2] = args.critic_hidden_size
        self.activation = activations[args.activation_critic.lower()]

        # critic 1
        self.bnorm_1 = nn.BatchNorm1d(args.state_dim+args.action_dim)
        self.lin1_1 = nn.Linear(args.state_dim+args.action_dim, l1)
        self.lnorm1_1 = LayerNorm(l1)
        self.lin2_1 = nn.Linear(l1, l2)
        self.lnorm2_1 = LayerNorm(l2)
        self.lout_1 = nn.Linear(l2, 1)

        # critic 2
        self.bnorm_2 = nn.BatchNorm1d(args.state_dim+args.action_dim)
        self.lin1_2 = nn.Linear(args.state_dim+args.action_dim, l1)
        self.lnorm1_2 = LayerNorm(l1)
        self.lin2_2 = nn.Linear(l1, l2)
        self.lnorm2_2 = LayerNorm(l2)
        self.lout_2 = nn.Linear(l2, 1)

        # initialize the weights with smaller values:
        self.lout_1.weight.data.mul_(0.1)
        self.lout_1.bias.data.mul_(0.1)
        self.lout_2.weight.data.mul_(0.1)
        self.lout_2.bias.data.mul_(0.1)

        self.to(args.device)

    def forward(self, state, action):
        # ----- critic 1 -----
        td_input = torch.cat((state, action), 1)
        td_input = self.bnorm_1(td_input)

        out = self.lin1_1(td_input)  # hidden layer 1_1
        out = self.lnorm1_1(out)
        out = self.activation(out)
        out = self.lin2_1(out)  # hidden layer 2_1
        out = self.lnorm2_1(out)
        out = self.activation(out)

        out1 = self.lout_1(out)  # output interface:

        # ----- critic 2 -----
        td_input = torch.cat((state, action), 1)
        td_input = self.bnorm_2(td_input)

        out = self.lin1_2(td_input)  # hidden layer 1_2
        out = self.lnorm1_2(out)
        out = self.activation(out)
        out = self.lin2_2(out)  # hidden layer 2_2
        out = self.lnorm2_2(out)
        out = self.activation(out)

        out2 = self.lout_2(out)  # output interface:
        # TODO: remove or add Tanh at the end of the critic network (-1,1)

        return out1, out2


# class TD3(object):
#     def __init__(self, args: Parameters):
#         self.args = args
#         self.buffer = ReplayMemory(args.individual_bs, args.device)
#         self.critical_buffer = ReplayMemory(args.individual_bs, args.device)

#         # initialise actor:
#         self.actor = Actor(args, init=True)
#         self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)
#         self.actor_lr_scheduler = ExponentialLR(self.actor_optim, gamma=0.999)

#         self.actor_target = Actor(args, init=True)

#         # initialise the critics:
#         self.critic = Critic(args)
#         self.critic_target = Critic(args)
#         self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
#         self.critic_lr_scheduler = ExponentialLR(
#             self.critic_optim, gamma=0.999)

#         # initialize the loss:
#         self.gamma = args.gamma
#         self.tau = args.tau

#         # initialize the target networks with the same weights as the original networks:
#         hard_update(self.actor_target, self.actor)
#         hard_update(self.critic_target, self.critic)

#         # Conditional for Action Policy Smoothness (CAPS)
#         self.caps_dict: dict = None

#         if self.args.use_caps:
#             self.caps_dict = {
#                 'lambda_s': 0.5,
#                 'lambda_t': 0.1,
#                 'eps_sd': 0.05,
#             }

#     def update_parameters(self, batch, iteration: int, champion_policy=False):
#         state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

#         with torch.no_grad():
#             self.actor_target.to(self.args.device)
#             self.critic_target.to(self.args.device)
#             self.critic.to(self.args.device)
#             state_batch = state_batch.to(self.args.device)
#             next_state_batch = next_state_batch.to(self.args.device)
#             action_batch = action_batch.to(self.args.device)
#             reward_batch = reward_batch.to(self.args.device)
#             done_batch = done_batch.to(self.args.device)

#             # select target action:
#             noise = (torch.randn_like(action_batch) *
#                      self.args.noise_sd).clamp(-self.args.noise_clip, self.args.noise_clip)
#             next_action_batch = torch.clamp(
#                 self.actor_target.forward(next_state_batch) + noise, -1, 1)

#             # compute the target Q values:
#             target_Q1, target_Q2 = self.critic_target.forward(
#                 next_state_batch, next_action_batch)
#             next_Q = torch.min(target_Q1, target_Q2) * (1-done_batch)
#             target_q = reward_batch + (self.gamma * next_Q).detach()

#         # get the current Q estimates:
#         current_q1, current_q2 = self.critic.forward(state_batch, action_batch)

#         # compute critic losses:
#         loss_q1 = F.mse_loss(current_q1, target_q)
#         loss_q2 = F.mse_loss(current_q2, target_q)
#         TD_loss = loss_q1 + loss_q2

#         # optimize the critics:
#         self.critic_optim.zero_grad()
#         TD_loss.backward()
#         nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
#         self.critic_optim.step()
#         self.critic_lr_scheduler.step()
#         TD_data = TD_loss.data.cpu().numpy()

#         # soft update the target networks:
#         pgl = None
#         if iteration % self.args.policy_update_freq == 0:
#             policy_grad_loss = self.actor_update(state_batch, action_batch)

#             if not champion_policy:
#                 soft_update(self.actor_target, self.actor, self.tau)

#             soft_update(self.critic_target, self.critic, self.tau)
#             pgl = policy_grad_loss.data.cpu().numpy()
#         return pgl, TD_data

#     def actor_update(self, state_batch, action_batch):
#         self.actor_optim.zero_grad()

#         # retrieve values from the critics:
#         # objective reward:
#         est_q1, _ = self.critic.forward(
#             state_batch, self.actor.forward(state_batch))
#         policy_grad_loss = -torch.mean(est_q1)

#         if self.caps_dict is not None:
#             next_action_batch = self.actor.forward(state_batch)
#             state_bar = state_batch + \
#                 torch.randn_like(state_batch) * self.caps_dict['eps_sd']
#             action_bar = self.actor.forward(state_bar)
#             caps_loss = self.caps_dict['lambda_t'] * F.mse_loss(
#                 action_bar, next_action_batch) + self.caps_dict['lambda_s'] * F.mse_loss(action_batch, action_bar)

#             policy_grad_loss += caps_loss

#         # backpropagate the policy gradient loss:
#         policy_grad_loss.backward()
#         nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
#         self.actor_optim.step()
#         self.actor_lr_scheduler.step()
#         return policy_grad_loss


class TD3ES(object):
    def __init__(self, args: ESParameters):
        self.args = args

        # initialise actor:
        if args.use_multiAgent:
            self.actor = MultiAgentActor(args, init=True)

        else:
            self.actor = CONV_ACTOR(args).to(args.device) if args.use_state_history else Actor(
                args, init=True).to(args.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        # initialise the critics:
        self.critic = CONV_CRITIC(args).to(
            args.device) if args.use_state_history else Critic(args).to(args.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        # initialize the loss:
        self.gamma = args.gamma
        self.tau = args.tau

        # initialize the target networks with the same weights as the original networks:
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Conditional for Action Policy Smoothness (CAPS)
        self.caps_dict: dict = None

        if self.args.use_caps:
            self.caps_dict = {
                'lambda_s': 0.5,
                'lambda_t': 0.1,
                'eps_sd': 0.05,
            }

    def update_parameters(self, batch, iteration: int, champion_policy=False):
        # self.actor = torch.compile(self.actor)
        # self.actor_target = torch.compile(self.actor_target)
        # self.critic = torch.compile(self.critic)
        # self.critic_target = torch.compile(self.critic_target)

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        with torch.no_grad():

            state_batch = state_batch.to(self.args.device)
            next_state_batch = next_state_batch.to(self.args.device)
            action_batch = action_batch.to(self.args.device)
            reward_batch = reward_batch.to(self.args.device)
            done_batch = done_batch.to(self.args.device)

            # select target action:
            noise = (torch.randn_like(action_batch) *
                     self.args.noise_sd).clamp(-self.args.noise_clip, self.args.noise_clip)
            next_action_batch = torch.clamp(
                self.actor_target.forward(next_state_batch) + noise, -1, 1)

            # compute the target Q values:
            target_Q1, target_Q2 = self.critic_target.forward(
                next_state_batch, next_action_batch)
            next_Q = torch.min(target_Q1, target_Q2) * (1-done_batch)
            target_q = reward_batch + (self.gamma * next_Q).detach()

        # get the current Q estimates:
        current_q1, current_q2 = self.critic.forward(state_batch, action_batch)

        # compute critic losses:
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)
        TD_loss = loss_q1 + loss_q2

        # optimize the critics:
        self.critic_optim.zero_grad()
        TD_loss.backward(retain_graph=True)
        # nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.critic_optim.step()
        TD_data = TD_loss.data.cpu().numpy()

        # soft update the target networks:
        pgl = None

        if iteration % self.args.policy_update_freq == 0:
            policy_grad_loss = self.actor_update(state_batch, action_batch)
            pgl = policy_grad_loss.data.cpu().numpy()

            # update the frozen target models
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        return pgl, TD_data

    def actor_update(self, state_batch, action_batch):

        # compute the actor loss
        est_q1, _ = self.critic.forward(
            state_batch, self.actor.forward(state_batch))
        policy_grad_loss = -torch.mean(est_q1)  # TODO: from - to +

        if self.caps_dict is not None:
            next_action_batch = self.actor.forward(state_batch)
            state_bar = state_batch + \
                torch.randn_like(state_batch) * self.caps_dict['eps_sd']
            action_bar = self.actor.forward(state_bar)
            caps_loss = self.caps_dict['lambda_t'] * F.mse_loss(
                action_bar, next_action_batch) + self.caps_dict['lambda_s'] * F.mse_loss(action_batch, action_bar)

            policy_grad_loss += caps_loss

        # back-propagate the policy gradient loss:
        self.actor_optim.zero_grad()
        policy_grad_loss.backward()
        # nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.actor_optim.step()

        return policy_grad_loss
