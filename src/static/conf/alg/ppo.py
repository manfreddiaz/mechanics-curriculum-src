import functools

import gym
from omegaconf import DictConfig

from ccgm.common.algs.ppo import PPO


def make_alg(
    id: str,
    hparams: DictConfig,
    rparams: DictConfig
):
    def make_ppo(
        envs: gym.vector.VectorEnv
    ):
        return functools.partial(
            PPO.learn,
            envs=envs,
            hparams=hparams,
            rparams=rparams
        )

    return make_ppo
