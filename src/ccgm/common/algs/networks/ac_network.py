import torch.nn as nn

from torch.distributions.categorical import Categorical
from .p_networks import MinAtarPolicy, MlpPolicy
from .v_networks import MinatarValueNetwork, MlpValueNetwork


class MlpActorCritic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = MlpPolicy(envs)
        self.critic = MlpValueNetwork(envs)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)  # TODO: potential seeding problem
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def predict(self, x):
        action, _, _, _ = self.get_action_and_value(x)
        return action


class MinatarActorCritic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = MinAtarPolicy(envs)
        self.critic = MinatarValueNetwork(envs)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def predict(self, x):
        action, _, _, _ = self.get_action_and_value(x)
        return action


class MiniGridActorCritic(nn.Module):

    def __init__(self, envs) -> None:
        super().__init__()
        in_channels = envs.single_observation_space.shape[-1]
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.embedding_size = 64
        self.actor =  nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, envs.single_action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_value(self, x):
        x = x.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.view(x.size(0), -1)
        return self.critic(x)
        

    def get_action_and_value(self, x, action=None):
        x = x.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.view(x.size(0), -1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def predict(self, x):
        action, _, _, _ = self.get_action_and_value(x)
        return action
