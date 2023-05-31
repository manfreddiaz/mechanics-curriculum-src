import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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


class MinAtarPolicy(nn.Module):

    def __init__(self, envs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            envs.single_observation_space.shape[0], 
            16, kernel_size=3, 
            stride=1
        )

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Sequential(
            layer_init(nn.Linear(num_linear_units, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def forward(self, x):
        # TODO: check this arch.
        x = F.relu(self.conv(x))
        x = self.fc_hidden(x.view(x.size(0), -1))
        return x

