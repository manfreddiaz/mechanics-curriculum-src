import numpy as np
import gym
import gym.spaces as spaces
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as f

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class StableBaselinesCompatWrapper(gym.ObservationWrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=tuple(reversed(env.observation_space.shape)), 
            dtype=np.uint8
        )

    def observation(self, observation: np.array):
        obs = observation.astype(np.uint8) * 255
        return obs.transpose(2, 0, 1)


class MinAtarFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        in_channels = observation_space.shape[0]
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=features_dim)

    def forward(self, x):
        x = f.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        return x

