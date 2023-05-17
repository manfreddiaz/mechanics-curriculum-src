import functools
import os
from typing import List, Union
import gym
import gym_minigrid

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


def make_vectorized_env(
    env_id,
    num_envs: int
):
    make_sync_env = functools.partial(
        make_minigrid_env,
        env_id=env_id,
    )
    return gym.vector.SyncVectorEnv(
        [make_sync_env for _ in range(num_envs)])


def make_task(
    id: str,
    order: str,
    num_envs: int,
    episode_limit: int
):
    def make_env_fn():
        return [
            make_vectorized_env(
                env_id=env_id,
                num_envs=num_envs
            ) for env_id in MINIGRIDS_GRIDS
        ]
    return None, make_env_fn
