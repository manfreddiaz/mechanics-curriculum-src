from typing import List
import gym
from stable_baselines3 import PPO
from ccgm.common.coalitions import Coalition, OrderedCoalition
from ccgm.common.games import CooperativeMetaGame


def make_coalition(
    team, 
    ordered,
    total_time_limit: int,  
    probs = None,
):

    if not ordered:
        return Coalition(
            strategies=[
                gym.make(strategy_id) for strategy_id in team
            ],
            probs=probs
        )
    else:
        return OrderedCoalition(
            strategies=[
                gym.make(strategy_id) for strategy_id in team
            ],
            probs=probs,
            time_limit=total_time_limit
        )


def make_cooperative_env(
    team: List,
    ordered: bool = False,
    probs: List = None,
    num_episodes: int = 500 
) -> 'CooperativeMetaGame':

    env = CooperativeMetaGame(
        meta_strategy=make_coalition(
            team=team,
            ordered=ordered,
            total_time_limit=num_episodes,
            probs=probs,
        )
    )

    return env

def make_policy():
    return "MlpPolicy"


def make_ppo(env, seed, args) -> 'PPO':
    return PPO(
        policy=make_policy(),
        env=env,
        verbose=0,
        seed=seed,
        n_steps=min(2048, args.episode_limit * args.num_episodes),
        device='cpu'
    )


def make_task(team, args):
    env = make_cooperative_env(
        team,
        ordered=args.ordered,
        episode_time_limit=args.episode_limit,
        num_episodes=args.num_episodes
    )

    algorithm = make_ppo(env, seed, args)

    return env, algorithm   


# class QNetwork(nn.Module):
#     def __init__(self, in_channels, num_actions):

#         super(QNetwork, self).__init__()

#         # One hidden 2D convolution layer:
#         #   in_channels: variable
#         #   out_channels: 16
#         #   kernel_size: 3 of a 3x3 filter matrix
#         #   stride: 1
#         self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

#         # Final fully connected hidden layer:
#         #   the number of linear unit depends on the output of the conv
#         #   the output consist 128 rectified units
#         def size_linear_unit(size, kernel_size=3, stride=1):
#             return (size - (kernel_size - 1) - 1) // stride + 1
#         num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
#         self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

#         # Output layer:
#         self.output = nn.Linear(in_features=128, out_features=num_actions)

#     # As per implementation instructions according to pytorch, the forward function should be overwritten by all
#     # subclasses
#     def forward(self, x):
#         # Rectified output from the first conv layer
#         x = f.relu(self.conv(x))

#         # Rectified output from the final hidden layer
#         x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

#         # Returns the output from the fully-connected linear layer
#         return self.output(x)