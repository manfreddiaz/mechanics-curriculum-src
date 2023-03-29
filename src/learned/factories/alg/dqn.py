

import functools
from omegaconf import DictConfig
import torch
from ccgm.common.algs.dqn import DQN


def make_alg(
    id: str,
    hparams: DictConfig,
    rparams: DictConfig
):
    def make_dqn(
        device: torch.device,
        logger,
        log_every: int,
        log_file_format: str
    ):
        play_fn = functools.partial(
            DQN.play,
            hparams=hparams,
            rparams=rparams,
            device=device,
        )

        optimize_fn = functools.partial(
            DQN.optimize,
            hparams=hparams,
            rparams=rparams,
            device=device,
            logger=logger,
            log_every=log_every,
            log_file_format=log_file_format,
        )

        repeat_play_fn = functools.partial(
            DQN.repeat_play,
            hparams=hparams,
            rparams=rparams,
            device=device,
            logger=logger,
            log_every=log_every,
            log_file_format=log_file_format,
        )

        learn_fn = functools.partial(
            DQN.learn,
            hparams=hparams,
            rparams=rparams,
            device=device,
            logger=logger,
            log_every=log_every,
            log_file_format=log_file_format,
        )

        return play_fn, optimize_fn, repeat_play_fn, learn_fn

    return make_dqn
