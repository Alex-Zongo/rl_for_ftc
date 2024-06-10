
import torch
import numpy as np
from environments.config import select_env
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FDI(torch.nn.Module):
    def __init__(self, env, args):
        super(FDI, self).__init__()
        self.network = torch.nn.Sequential(
            layer_init(torch.nn.Linear(
                env.single_observation_space.shape[0]+env.single_action_space.shape[0], 64)),
            torch.nn.ReLU() if args.use_relu else torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.ReLU() if args.use_relu else torch.nn.Tanh(),
        )

        self.FD = torch.nn.Sequential(
            layer_init(torch.nn.Linear(64, 1)),
            # torch.nn.Sigmoid(),
        )
        self.KParams = torch.nn.Sequential(
            layer_init(torch.nn.Linear(64, 3)),
            torch.nn.Tanh(),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.network(x)
        FD = self.FD(x)
        KParams = self.KParams(x)
        return FD, KParams

    def binary(self, f):
        l = torch.sigmoid(f)
        pred = (l > 0.5).float()
        return pred

    def count_kp_parameters(self):
        return sum(p.numel() for p in self.KParams.parameters() if p.requires_grad)

    def extract_kparams(self):
        tot_size = self.count_kp_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32)
        count = 0
        for name, param in self.KParams.named_parameters():
            if len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count+sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    def inject_kparams(self, params):
        count = 0
        pvec = params.clone().detach()
        for name, param in self.KParams.named_parameters():
            if len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count+sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            param.requires_grad = True
            count += sz


def make_env(gym_id, seed, add_sm=False, eval=False, t_max=20):
    def thunk():
        if gym_id.startswith('PHlab'):
            env = select_env(
                environment_name=gym_id,
                conform_with_sb=True,
                add_sm_to_reward=add_sm
            )
            env.t_max = t_max  # setting t_max to 10s
            if eval:
                env.set_eval_mode(t_max=80)
        elif gym_id.startswith('LunarLander'):
            env = gym.make(
                gym_id,
                continuous=True,
                gravity=-9.8,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5,
            )
            # print(env.action_space)
        elif gym_id.startswith('Pendulum'):
            env = gym.make(
                gym_id,
                g=9.81,
            )
        elif gym_id.startswith('BipedalWalker'):
            env = gym.make(
                gym_id,
                # g=9.81,
            )
        else:
            env = gym.make(gym_id)

        # keep track of the episode cumulative reward and episode length (info['episode']['r'] and info['episode']['l'])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = Monitor(env)
        # env = ActionLoggingWrapper(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        check_env(env)
        return env

    return thunk
