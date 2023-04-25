from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from ccgm.common.envs.sl.cifar10.net import Net


def make_agent(
    id: str
):

    def agent_fn(
        envs: DataLoader,
        hparams: DictConfig,
        device: torch.device
    ):
        network = Net()
        network.to(device)
        return network

    return agent_fn
