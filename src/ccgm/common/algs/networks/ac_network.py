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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

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
