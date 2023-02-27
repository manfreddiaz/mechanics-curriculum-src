

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from ccgm.common.coalitions import Coalition


def make_xpt_dir(cfg):
    return os.path.join(
        cfg.run.outdir,
        f"{cfg.task.id}", 
        f"{cfg.task.order}",
        f"{cfg.alg.id}"
    )

def make_xpt_coalition_dir(
    training_coalition: Coalition,
    cfg: DictConfig
):
    
    return os.path.join(
        make_xpt_dir(cfg), 
        f'game-{str(training_coalition.idx)}'
    )

def hydra_load_node(x: str):
    cfg = hydra.compose(f"{x}.yaml")
    cfg = cfg[list(cfg.keys())[0]]
    return cfg

def hydra_custom_resolvers():
    OmegaConf.register_new_resolver(
        "bmult", lambda x, y: x * y
    )

    OmegaConf.register_new_resolver(
        "bdiv", lambda x, y: x // y
    )

    OmegaConf.register_new_resolver(
        "load", hydra_load_node
    )