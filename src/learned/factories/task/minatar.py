import functools
from typing import Union

import gym

from ccgm.common.envs.rl.gym.miniatar import (
    MINATAR_STRATEGIES_all, MINATAR_STRATEGIES_v0,
    MINATAR_STRATEGIES_v1
)
from ccgm.common.envs.rl.gym.miniatar.utils import MinAtarStandardObservation


def make_minatar_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    env = MinAtarStandardObservation(env)
    return env


def make_vectorized_env(
    env_id,
    num_envs: int
):
    make_sync_env = functools.partial(
        make_minatar_env,
        env_id=env_id,
    )
    return gym.vector.SyncVectorEnv(
        [make_sync_env for _ in range(num_envs)])


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
        raise NotImplementedError()

    # game_spec = CoalitionalGame.make(
    #     players=players,
    #     ordered=True if order == 'ordered' else False
    # )

    def make_env_fn():
        return [
            make_vectorized_env(
                env_id=env_id,
                num_envs=num_envs
            ) for env_id in players
        ]

    return None, make_env_fn
