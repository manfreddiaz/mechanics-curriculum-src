import functools
import os
from typing import List, Union
import gym
import gym_minigrid

from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from ccgm.common.coalitions import Coalition, OrderedCoalition
from ccgm.common.games import CooperativeMetaGame
from ccgm.utils import CoalitionalGame

from ccgm.common.envs.rl.gym.minigrid import (
  MINIGRIDS_GRIDS   
)


def make_minigrid_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = gym_minigrid.wrappers.OneHotPartialObsWrapper(env)
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    return env

def make_coalition(
    team, 
    order,
    total_time_limit: int,  
    probs = None,
):

    if order == 'random':
        return Coalition(
            players=[
                make_minigrid_env(player) for player in team.players
            ],
            probs=probs
        )
    elif order == 'ordered':
        return OrderedCoalition(
            players=[
                make_minigrid_env(player) for player in team.players
            ],
            probs=probs,
            time_limit=total_time_limit
        )
    else:
        raise NotImplementedError()


def make_cooperative_env(
    team: List,
    episode_time_limit: int,
    order: str,
    probs: List = None,

) -> 'CooperativeMetaGame':

    return CooperativeMetaGame(
        meta_strategy=make_coalition(
            team=team,
            order=order,
            probs=probs,
            total_time_limit=episode_time_limit
        )
    )


def make_task(
    id: str,
    order: str,
    num_envs: int,
    episode_limit: int
):
    game_spec = CoalitionalGame.make(
        players=MINIGRIDS_GRIDS,
        ordered=True if order == 'ordered' else False
    )

    def make_env(
        team: List[int], probs: list[int], 
        team_dir: str, seed: int, train: bool = True
    ):
        def monitored():
            return Monitor(
                make_cooperative_env(
                    team=team,
                    order=order,
                    probs=probs,
                    episode_time_limit=episode_limit,
                ),
                filename=os.path.join(team_dir, f'{seed}.train' if train else f'{seed}.eval'),
                info_keywords=('meta-strategy',)
            )
        envs = gym.vector.SyncVectorEnv([
            monitored for i in range(num_envs)
        ])
        envs.seed(seed)
        return envs

    return game_spec, make_env
