
import functools

import hydra
import torch
from omegaconf import DictConfig

from learned.core import MetaTrainingEnvironment
from learned.utils import hydra_custom_resolvers


hydra_custom_resolvers()


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:

    make_agent = hydra.utils.instantiate(cfg.agent)
    _, make_env = hydra.utils.instantiate(cfg.task)
    make_alg = hydra.utils.instantiate(cfg.alg)

    env = MetaTrainingEnvironment(
        agent_fn=functools.partial(
            make_agent,
            hparams=cfg.alg.hparams,
            device=torch.device(cfg.torch.device)
        ),
        env_fn=make_env,
        alg_fn=make_alg,
        eval_fn=None
    )


if __name__ == '__main__':
    main()
