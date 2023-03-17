import functools

import gym

from ccgm.common.envs.sgt.env import make_env
from ccgm.common.envs.sgt.impl.prisioner_dilemma import SPID_STRATEGIES


def make_vectorized_env(
    env_id,
    episode_time_limit: int,
    sparse: bool,
    one_hot: bool,
    num_envs: int
):
    make_sync_env = functools.partial(
        make_env,
        env_id=env_id,
        episode_time_limit=episode_time_limit,
        sparse=sparse,
        one_hot=one_hot
    )
    return gym.vector.SyncVectorEnv(
        [make_sync_env for _ in range(num_envs)])


def make_task(
    id: str,
    order: str,
    num_episodes: int,
    num_envs: int,
    episode_limit: int,
    sparse: bool,
    one_hot: bool
):

    def make_env_fn():
        return [
            make_vectorized_env(
                env_id=env_id,
                episode_time_limit=episode_limit,
                sparse=sparse,
                one_hot=one_hot,
                num_envs=num_envs
            ) for env_id in SPID_STRATEGIES
        ]

    return None, make_env_fn
