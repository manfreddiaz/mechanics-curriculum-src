import functools

from omegaconf import DictConfig

from ccgm.common.algs.dqn import DQN


def make_alg(
    id: str,
    hparams: DictConfig,
    rparams: DictConfig
):
    def make_dqn(
        envs
    ):
        return functools.partial(
            DQN.learn,
            envs=envs,
            hparams=hparams,
            rparams=rparams
        )

    return make_dqn
