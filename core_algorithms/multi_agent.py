import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from core_algorithms.model_utils import activations, LayerNorm, hard_update, soft_update, is_lnorm_key
from core_algorithms.some_actor_model import LSTM_ACTOR, LSTM_CRITIC, CONV_ACTOR, CONV_CRITIC
from parameters import Parameters
from parameters_es import ESParameters
from core_algorithms.replay_memory import ReplayMemory


class SingleActor(nn.Module):
    def __init__(self, args: ESParameters, init=False):
        super(SingleActor, self).__init__()
        self.args = args
        h = int(self.args.actor_hidden_size/3)
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
            nn.Linear(h, 1),
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


class MultiAgentActor(nn.Module):
    def __init__(self, args: ESParameters, init=False):
        super(MultiAgentActor, self).__init__()
        self.args = args
        self.actor_elevator = SingleActor(args, init)
        self.actor_ailerons = SingleActor(args, init)
        self.actor_rudder = SingleActor(args, init)
        self.to(args.device)

    def forward(self, state: torch.tensor):
        elevator = self.actor_elevator(state)
        ailerons = self.actor_ailerons(state)
        rudder = self.actor_rudder(state)
        return torch.cat((elevator, ailerons, rudder), dim=1)

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
