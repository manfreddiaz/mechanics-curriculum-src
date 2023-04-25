import os
import random
import yaml
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.datasets import FashionMNIST

from ccgm.common.envs.sl.build import build
from ccgm.common.envs.sl.fashionmnist.config import ROOT_DIR
from ccgm.common.envs.sl.fashionmnist.net import Net
from ccgm.common.envs.sl.utils import (
    compute_confusion_matrix, compute_treachorus_pairs, train_or_load_model
)

def main(
    seed:int = 1234, batch_size: int = 4, 
    epochs: int = 200, cfm_on_val: bool = True,
    max_players: int = 6
):

    classes = (
        'tshirt-top', 'trouser', 'pullover', 'dress',
        'coat', 'sandal', 'shirt', 'sneaker', 
        'bag', 'ankle-boot'
    )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    build(
        dataset_name="fashion-mnist", dataset_fn=FashionMNIST, classes=classes,
        transform=transform, root_dir=ROOT_DIR, net=Net(),
        seed=seed, batch_size=batch_size, epochs=epochs, cfm_on_val=cfm_on_val,
        max_players=max_players
    )
   

if __name__ == '__main__':
    main()

