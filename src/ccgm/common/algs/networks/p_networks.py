import numpy as np
import torch.nn as nn

from .utils import layer_init


class MlpPolicy(nn.Module):

    def __init__(self, envs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    
    def forward(self, x):
        return self.mlp(x)
