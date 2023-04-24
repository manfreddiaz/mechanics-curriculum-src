import functools
import os
from typing import List, Union
import gym
from stable_baselines3.common.monitor import Monitor
from ccgm.common.coalitions import Coalition, OrderedCoalition
from ccgm.common.envs.rl.gym.miniatar.utils import MinAtarStandardObservation
from ccgm.common.games import CooperativeMetaGame
from ccgm.utils import CoalitionalGame

from ccgm.common.envs.rl.gym.miniatar import (
    MINATAR_STRATEGIES_v0,
    MINATAR_STRATEGIES_v1,
    MINATAR_STRATEGIES_all
)

def make_minatar_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = MinAtarStandardObservation(env)
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
                make_minatar_env(player) for player in team.players
            ],
            probs=probs
        )
    elif order == 'ordered':
        return OrderedCoalition(
            players=[
                make_minatar_env(player) for player in team.players
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
    version: Union[int, str],
    episode_limit: int
):
    if version == 0:
        players = MINATAR_STRATEGIES_v0
    elif version == 1:
        players = MINATAR_STRATEGIES_v1
    elif version == 'all':
        players = MINATAR_STRATEGIES_all
    else:
        raise ValueError(version)
    
    game_spec = CoalitionalGame.make(
        players=players,
        ordered=True if order == 'ordered' else False
    )

    def make_env(
        team: List[int], probs: list[int], 
        team_dir: str, seed: int
    ):
        def monitored():
            return Monitor(
                make_cooperative_env(
                    team=team,
                    order=order,
                    probs=probs,
                    episode_time_limit=episode_limit,
                ),
                filename=os.path.join(team_dir, f'{seed}.train'),
                info_keywords=('meta-strategy',)
            )
        envs = gym.vector.SyncVectorEnv([
            monitored for i in range(num_envs)
        ])
        envs.seed(seed)
        return envs

    return game_spec, make_env
