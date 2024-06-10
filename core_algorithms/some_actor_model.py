import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from parameters_es import ESParameters
from core_algorithms.model_utils import activations, is_lnorm_key, LayerNorm
from core_algorithms.replay_memory import ReplayMemory, PrioritizedReplayMemory


class RLNN(nn.Module):
    '''Base Class for all RL Neural Networks.'''

    def __init__(self, args: ESParameters):
        super(RLNN, self).__init__()
        self.args = args

    def extract_parameters(self):
        ''' Extract the parameters of the network and flatten it into a single vector.
        This is used for the genetic algorithm.

        Returns:
            torch.tensor: Flattened parameters of the network.
        '''
        tot_size = self.count_parameters()
        p_vec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        i = 0
        for name, param in self.named_parameters():
            if 'lnorm' in name or len(param.shape) != 2:
                continue
            sz = param.numel()
            p_vec[i:i+sz] = param.view(-1)
            i += sz
        return p_vec.detach().clone()

    def inject_parameters(self, parameters):
        ''' Inject the parameters into the network. This is used for the genetic algorithm.

        Args:
            parameters (torch.tensor): Flattened parameters of the network.
        '''
        i = 0
        parameters = torch.tensor(parameters)
        for name, param in self.named_parameters():
            if 'lnorm' in name or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = parameters[i:i+sz]
            reshaped = raw.reshape(param.shape)
            param.data.copy_(reshaped.data)
            i += sz

    def count_parameters(self):
        ''' Count the number of parameters in the network.'''
        count = 0
        for name, param in self.named_parameters():
            if 'lnorm' in name or len(param.shape) != 2:
                continue
            count += np.prod(param.shape)
        return count

    def extract_grads(self):
        """Returns the gradients of the network.

        Returns:
            _type_: _description_
        """
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


class Actor(RLNN):
    ''' Actor model class:

    This class is used to generate actions based on the current state of the environment.'''

    def __init__(self, args: ESParameters, init=False):
        super(Actor, self).__init__(args)
        if not init:
            return
        self.args = args
        self.h = self.args.actor_hidden_size
        self.L = self.args.actor_num_layers
        # self.buffer = ReplayMemory(self.args)
        # self.critical_buffer = ReplayMemory(self.args)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        layers = []
        self.activation = activations[args.activation_actor.lower()]
        # input layer:
        layers.extend([
            nn.Linear(args.state_dim, self.h),
            LayerNorm(self.h),
            self.activation,
        ])

        # hidden layer(s):
        for _ in range(self.L):  # TODO: L==1
            layers.extend([
                nn.Linear(self.h, self.h),
                LayerNorm(self.h),
                self.activation,
            ])

        # output layer:
        layers.extend([
            nn.Linear(self.h, args.action_dim),
            nn.Tanh(),
        ])
        self.net = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.net = self.net.to(args.device)

    def forward(self, state: torch.tensor):

        return self.net(state)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.args.device)
        return self.forward(state).cpu().data.numpy()

    def update(self, buffer: ReplayMemory, critic, actor_t, current_step):
        ''' Update the actor model based on the critic model.'''
        # sample from the replay buffer:
        states, _, _, _, _ = buffer.sample(
            self.args.batch_size)

        # compute actor loss:
        # TODO NOTE: use td3 by default:
        next_actions = self.forward(states)
        est_q1, _ = critic.forward(states, next_actions)
        actor_loss = -torch.mean(est_q1)

        # Optimize the actor:
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # update the frozen target models:
        if current_step % self.args.policy_freq == 0:
            for param, target_param in zip(self.parameters(), actor_t.parameters()):
                target_param.data.copy_(
                    self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


class CONV_ACTOR(RLNN):
    def __init__(self, args: ESParameters):
        super(CONV_ACTOR, self).__init__(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.h = args.actor_hidden_size
        self.L = args.actor_num_layers
        self.state_length = args.state_length
        self.input_dim = args.state_dim

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=2*self.h, kernel_size=(self.state_length, 1))
        self.conv2 = nn.Conv2d(
            in_channels=2*self.h, out_channels=self.h, kernel_size=(1, 1))
        self.conv_out = nn.Linear(self.h*self.input_dim, self.h)

        self.activation = nn.Tanh()

        layers = []
        if self.L >= 2:
            for _ in range(self.L):
                layers.extend([
                    nn.Linear(self.h, self.h),
                    LayerNorm(self.h),
                    self.activation
                ])

        layers.extend([
            nn.Linear(self.h, args.action_dim),
            self.activation
        ])
        self.args = args
        self.net = nn.Sequential(*layers)
        self.to(args.device)

    def forward(self, state: torch.tensor):

        B = state.shape[0] if state.ndim > 2 else 1
        # print(">> state shape: ", state.shape)
        state = state.to(self.args.device)
        s_out = torch.tanh(self.conv1(state.reshape(
            B, 1, self.state_length, -1)))
        s_out = torch.tanh(self.conv2(s_out))
        # print(">> CONV output shape: ", s_out.shape)
        s_out = torch.tanh(self.conv_out(s_out.view(B, -1)))
        # print(">> FC output shape: ", s_out.shape)
        a_out = self.net(s_out.reshape(B, -1))
        return a_out

    def select_action(self, state: torch.tensor):
        # print(">> State DIM: ", state.ndim)
        if state.ndim > 2:
            B, L, _ = state.shape
            state = torch.FloatTensor(state.reshape(B, L, -1))
        elif state.ndim == 2:
            L, _ = state.shape
            state = torch.FloatTensor(state.reshape(1, L, -1))
        else:
            state = torch.FloatTensor(state.reshape(1, -1))
        return self.forward(state).cpu().data.numpy().flatten()


class CONV_CRITIC(RLNN):
    def __init__(self, args):
        super(CONV_CRITIC, self).__init__(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.h = args.critic_hidden_size[0]
        self.state_length = args.state_length
        self.input_dim = args.state_dim
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=2*self.h, kernel_size=(self.state_length, 1))
        self.conv2 = nn.Conv2d(
            in_channels=2*self.h, out_channels=self.h, kernel_size=(1, 1))
        self.state_enc = nn.Linear(self.h*self.input_dim, self.h)
        self.activation = nn.Tanh()

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

        # initialize the weights with smaller values:
        self.linear1.weight.data.mul_(0.1)
        self.linear1.bias.data.mul_(0.1)
        self.linear11.weight.data.mul_(0.1)
        self.linear11.bias.data.mul_(0.1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.mul_(0.1)
        self.linear22.weight.data.mul_(0.1)

        self.to(args.device)

    def forward(self, state, action):
        B = state.shape[0] if state.ndim > 2 else 1
        s_out = torch.tanh(self.conv1(state.reshape(
            B, 1, self.state_length, -1)))
        s_out = torch.tanh(self.conv2(s_out))
        # print(">> CONV output shape: ", s_out.shape)
        s_out = torch.tanh(self.state_enc(s_out.view(B, -1)))

        # critic 1:
        out = torch.cat((s_out.reshape(B, -1), action), dim=1)
        out1 = self.bn1(out)
        out1 = self.ln1(self.linear11(out1))
        out1 = self.activation(out1)
        out1 = self.linear1(out1)

        # critic 2:
        out2 = self.bn2(out)
        out2 = self.ln2(self.linear22(out2))
        out2 = self.activation(out2)
        out2 = self.linear2(out2)

        return out1, out2

# Other actor models
# TODO:
# 1. Implement RNN_Actor or LSTM NN
# 2. Implement a Transformer model type:


class LSTM_ACTOR(RLNN):
    def __init__(self, args, device, input_size, hidden_size, num_layers, state_length):
        super(LSTM_ACTOR, self).__init__(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.sequence_length = state_length
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(state_length, 1))
        self.linear = nn.Linear(hidden_size, 3)
        self.activation = nn.Tanh()
        self.device = device

        self.to(device)

    def forward(self, state: torch.tensor):
        state = state.to(self.device)
        B = state.shape[0] if state.ndim > 2 else 1
        # print(">> state shape: ", state.shape)
        s_out, _ = self.lstm(state)

        # TODO add a relu or tanh function after the lstm network
        s_out = F.tanh(s_out)
        s_out = self.conv(s_out.reshape(
            B, 1, self.sequence_length, -1))  # B, 1, 1, h
        a_out = self.linear(F.tanh(s_out.reshape(B, -1)))
        return self.activation(a_out)

    def select_action(self, state: torch.tensor):
        # print(">> State DIM: ", state.ndim)
        if state.ndim > 2:
            B, L, _ = state.shape
            state = torch.FloatTensor(state.reshape(B, L, -1))
        elif state.ndim == 2:
            L, _ = state.shape
            state = torch.FloatTensor(state.reshape(1, L, -1))
        else:
            state = torch.FloatTensor(state.reshape(1, -1))
        return self.forward(state).cpu().data.numpy().flatten()


class LSTM_CRITIC(RLNN):
    def __init__(self, args, device, input_size, action_dim, hidden_size, num_layers, state_length, **kwargs):
        super(LSTM_CRITIC, self).__init__(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(state_length, 1))
        self.sequence_length = state_length
        self.activation = nn.Tanh()
        # critic 1
        self.bn1 = nn.BatchNorm1d(hidden_size + action_dim)
        self.linear11 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.linear1 = nn.Linear(hidden_size, 1)
        self.ln1 = LayerNorm(hidden_size)

        # critic 2
        self.bn2 = nn.BatchNorm1d(hidden_size + action_dim)
        self.linear22 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.ln2 = LayerNorm(hidden_size)

        # initialize the weights with smaller values:
        self.linear1.weight.data.mul_(0.1)
        self.linear1.bias.data.mul_(0.1)
        self.linear11.weight.data.mul_(0.1)
        self.linear11.bias.data.mul_(0.1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.mul_(0.1)
        self.linear22.weight.data.mul_(0.1)
        self.device = device
        self.to(device)

    def forward(self, state, action):
        state = state.to(self.device)
        B = state.shape[0] if state.ndim > 2 else 1
        s_out, _ = self.lstm(state)
        # B, 1, 1, h.
        s_out = F.tanh(s_out)
        s_out = F.tanh(self.conv(s_out.reshape(
            B, 1, self.sequence_length, -1)))
        # print(">> Output from LSTM: ", s_out.shape)
        # critic 1:
        out = torch.cat((s_out.reshape(B, -1), action), dim=1)
        out1 = self.bn1(out)
        out1 = self.ln1(self.linear11(out1))
        out1 = self.activation(out1)
        out1 = self.linear1(out1)

        # critic 2:
        # out2 = torch.cat((s_out[:, -1, :], action), dim=1)
        out2 = self.bn2(out)
        out2 = self.ln2(self.linear22(out2))
        out2 = self.activation(out2)
        out2 = self.linear2(out2)

        return out1, out2


class RNN_Actor(RLNN):
    def __init__(self, args: ESParameters, rnn_type='LSTM', init=True):
        if not init:
            return
        super(RNN_Actor, self).__init__(args)
        self.args = args
        self.h = self.args.actor_hidden_size
        self.L = self.args.actor_num_layers
        self.activation = activations[self.args.activation_actor.lower()]
        # self.critical_buffer = ReplayMemory(self.args)

        in_layer = []
        # input layer:
        in_layer.extend([
            nn.Linear(self.args.state_dim, self.h),
            LayerNorm(self.h),
            self.activation,
        ])

        # hidden RNN layers: #TODO: L==1
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.h, self.h, num_layers=self.L)
        else:
            self.rnn = nn.RNN(self.h, self.h, num_layers=self.L)

        # output layer:
        out_layer = []
        out_layer.extend([
            nn.Linear(self.h, self.args.action_dim),
            nn.Tanh(),
        ])
        self.in_net = nn.Sequential(*in_layer)
        self.ou_net = nn.Sequential(*out_layer)
        self.to(self.args.device)

    def forward(self, state: torch.tensor):
        h = self.in_net(state)
        out, h = self.rnn(h)
        return self.ou_net(out)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
        return self.forward(state).cpu().data.numpy()

    def update(self, buffer: ReplayMemory, critic, actor_t, current_step):
        ''' Update the actor model based on the critic model.'''

        # update the frozen target models:
        if current_step % self.args.policy_freq == 0:
            # sample from the replay buffer:
            states, _, _, _, _ = buffer.sample(
                self.args.batch_size)

            # compute actor loss:
            # TODO NOTE: use td3 by default:
            actor_loss = -critic(states, self(states))[0].mean()

            # Optimize the actor:
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

            for param, target_param in zip(self.parameters(), actor_t.parameters()):
                target_param.data.copy_(
                    self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


class Critic(RLNN):
    def __init__(self, args: ESParameters):
        super(Critic, self).__init__(args)
        self.args = args

        self.h1, self.h2 = self.args.critic_hidden_size
        self.L = self.args.critic_num_layers
        self.activation = activations[self.args.activation_critic.lower()]
        layers = []
        # input layer:
        layers.extend([
            nn.Linear(self.args.state_dim+self.args.action_dim, self.h1),
            LayerNorm(self.h1),
            self.activation,
        ])

        # hidden layers:
        for _ in range(self.L):  # TODO: L==1
            layers.extend([
                nn.Linear(self.h1, self.h2),
                LayerNorm(self.h2),
                self.activation,
            ])

        # output:
        layers.extend([
            nn.Linear(self.h2, 1),
        ])

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.critic_lr)

        self.net = nn.Sequential(*layers)
        self.to(self.args.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class CriticTD3(RLNN):
    def __init__(self, args: ESParameters):
        super(CriticTD3, self).__init__(args)
        self.args = args
        self.h1, self.h2 = args.critic_hidden_size
        self.L = args.critic_num_layers
        self.activation = activations[self.args.activation_critic.lower()]
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # Q1 architecture:
        self.layers1 = [
            nn.BatchNorm1d(self.args.state_dim+self.args.action_dim),
            nn.Linear(self.args.state_dim+self.args.action_dim, self.h1),
            LayerNorm(self.h1),
            self.activation,
            nn.Linear(self.h1, self.h2),
            LayerNorm(self.h2),
            self.activation,
        ]
        lout_1 = nn.Linear(self.h2, 1)
        lout_1.weight.data.mul_(0.1)
        lout_1.bias.data.mul_(0.1)

        self.layers1.extend([
            lout_1,
        ])

        self.layers1 = nn.Sequential(*self.layers1).to(self.args.device)

        # Q2 architecture:
        self.layers2 = [
            nn.BatchNorm1d(self.args.state_dim+self.args.action_dim),
            nn.Linear(self.args.state_dim+self.args.action_dim, self.h1),
            LayerNorm(self.h1),
            self.activation,
            nn.Linear(self.h1, self.h2),
            LayerNorm(self.h2),
            self.activation,
        ]
        lout_2 = nn.Linear(self.h2, 1)
        lout_2.weight.data.mul_(0.1)
        lout_2.bias.data.mul_(0.1)

        self.layers2.extend([
            lout_2,
        ])

        self.layers2 = nn.Sequential(*self.layers2).to(self.args.device)

        # self.buffer = ReplayMemory(args)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)

    def forward(self, state, action):
        x = torch.cat([torch.FloatTensor(state),
                      torch.FloatTensor(action)], dim=1)
        x = x.to(self.args.device)
        out1 = self.layers1(x)
        out2 = self.layers2(x)
        return out1, out2

    def update(self, buffer: ReplayMemory, actor_t, critic_t, current_step):
        # sample from the buffer:
        states, actions, next_states, rewards, done = buffer.sample(
            self.args.batch_size)

        # select action according to policy
        next_actions = actor_t.select_action(states)
        noise = np.clip(np.random.normal(
            0, self.args.policy_noise, size=(self.args.batch_size, self.args.action_dim)), -self.args.noise_clip, self.args.noise_clip)
        next_actions = np.clip(
            next_actions + noise, -1.0, 1.0)

        with torch.no_grad():
            # compute the target Q value
            rewards = rewards.to(self.args.device)
            done = done.to(self.args.device)
            target_Q1, target_Q2 = critic_t(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2).to(self.args.device)
            target_Q = rewards + (1-done) * self.args.gamma * target_Q

        # get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss:
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # update the targets:
        if current_step % self.args.policy_freq == 0:
            for param, target_param in zip(self.parameters(), critic_t.parameters()):
                target_param.data.copy_(
                    self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
