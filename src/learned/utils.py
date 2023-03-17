
import hydra
from omegaconf import OmegaConf


def _hydra_load_node(x: str):
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
        "load", _hydra_load_node
    )
