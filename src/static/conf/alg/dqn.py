

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
        alg = DQN(
            envs=envs,
            rparams=rparams,
            hparams=hparams
        )
        return alg

    return make_dqn