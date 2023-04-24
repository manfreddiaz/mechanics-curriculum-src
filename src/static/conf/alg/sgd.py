
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import functools
from ccgm.common.envs.sl.utils import train_or_load_model


def make_alg(
    id: str,
    optim: str,
    hparams: DictConfig,
):
    assert optim == "adam"

    def alg_fn(envs: DataLoader):
        return functools.partial(
            train_or_load_model,
            trainloader=envs,
            epochs=hparams.epochs,
            save_path=None,
            learning_rate=hparams.learning_rate
        )

    return alg_fn