import functools
from omegaconf import DictConfig
import torch

from ccgm.common.algs.ppo import PPO


def make_alg(
    id: str,
    hparams: DictConfig,
    rparams: DictConfig
):
    def make_ppo(
        device: torch.device,
        logger,
        log_every: int,
        log_file_format: str
    ):
        play_fn = functools.partial(
            PPO.play,
            hparams=hparams,
            rparams=rparams,
            device=device,
            logger=logger,
            log_every=log_every,
            log_file_format=log_file_format,
        )

        optimize_fn = functools.partial(
            PPO.optimize,
            hparams=hparams,
            rparams=rparams,
            device=device,
            logger=logger,
            log_every=log_every,
            log_file_format=log_file_format,
        )

        return play_fn, optimize_fn

    return make_ppo
