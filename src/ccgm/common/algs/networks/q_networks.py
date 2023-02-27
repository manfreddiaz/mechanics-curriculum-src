import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class MinAtarQNetwork(nn.Module):
    # https://github.com/kenjyoung/MinAtar/blob/master/examples/dqn.py
    def __init__(self, envs):

        super(MinAtarQNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(
            envs.single_observation_space.shape[0], 
            16, kernel_size=3, 
            stride=1
        )

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=envs.single_action_space.n)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)
    
    def predict(self, x):
        q_values = self(x)
        return torch.argmax(q_values, dim=1)
