import os
from typing import List
import gym
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from ccgm.common.coalitions import Coalition, OrderedCoalition
from ccgm.common.envs.sgt.impl.prisioner_dilemma import SPID_STRATEGIES
from ccgm.common.envs.sgt.wrappers import (
    OneHotObservationWrapper,
    SparseRewardWrapper
)
from ccgm.common.games import CooperativeMetaGame
from ccgm.utils import CoalitionalGame


def make_env(env_id, episode_time_limit, sparse, one_hot):
    env = gym.make(env_id)
    env = TimeLimit(
        env,
        episode_time_limit
    )
    if sparse:
        env = SparseRewardWrapper(env)

    if one_hot:
        env = OneHotObservationWrapper(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env


def make_coalition(
    team: List,
    order: str,
    episode_time_limit: int,
    total_time_limit: int,
    probs: List[float] = None,
    sparse: bool = True,
    one_hot: bool = True
):

    if order == 'random':
        return Coalition(
            players=[
                make_env(
                    env_id=player,
                    episode_time_limit=episode_time_limit,
                    sparse=sparse,
                    one_hot=one_hot
                ) for player in team.players
            ],
            probs=probs
        )
    elif order == 'ordered':
        return OrderedCoalition(
            players=[
                make_env(
                    env_id=player,
                    episode_time_limit=episode_time_limit,
                    sparse=sparse,
                    one_hot=one_hot
                ) for player in team.players
            ],
            probs=probs,
            time_limit=total_time_limit
        )
    else:
        raise NotImplementedError()


def make_cooperative_env(
    team: List,
    ordered: bool = False,
    probs: List = None,
    episode_time_limit: int = 200,
    num_episodes: int = 500,
    sparse: bool = True,
    one_hot: bool = True, 
) -> 'CooperativeMetaGame':

    env = CooperativeMetaGame(
        meta_strategy=make_coalition(
            team=team,
            order=ordered,
            episode_time_limit=episode_time_limit,
            total_time_limit=num_episodes,
            probs=probs,
            sparse=sparse,
            one_hot=one_hot
        )
    )

    return env


def make_task(
    id: str,
    order: str,
    num_episodes: int,
    num_envs: int,
    episode_limit: int,
    sparse: bool,
    one_hot: bool
):
    game_spec = CoalitionalGame.make(
        players=SPID_STRATEGIES,
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
                    ordered=order,
                    probs=probs,
                    episode_time_limit=episode_limit,
                    num_episodes=num_episodes,
                    sparse=sparse,
                    one_hot=one_hot
                ),
                filename=os.path.join(
                    team_dir, f'{seed}.train' if train else f'{seed}.eval'),
                info_keywords=('meta-strategy',)
            )
        envs = gym.vector.SyncVectorEnv([
            monitored for i in range(num_envs)
        ])
        envs.seed(seed)
        return envs

    return game_spec, make_env
