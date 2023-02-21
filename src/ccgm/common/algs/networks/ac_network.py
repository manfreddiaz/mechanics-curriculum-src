import torch.nn as nn

from torch.distributions.categorical import Categorical
from .p_networks import MlpPolicy
from .v_networks import MlpValueNetwork

class MlpActorCritic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.actor = MlpPolicy(envs)
        self.critic = MlpValueNetwork(envs)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
