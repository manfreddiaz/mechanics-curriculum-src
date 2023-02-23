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
