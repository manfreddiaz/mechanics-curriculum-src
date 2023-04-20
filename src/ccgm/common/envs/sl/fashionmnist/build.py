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

from ccgm.common.envs.sl.fashionmnist.config import ROOT_DIR
from ccgm.common.envs.sl.fashionmnist.net import Net
from ccgm.common.envs.sl.utils import (
    compute_confusion_matrix, compute_treachorus_pairs, train_or_load_model
)

def main(
    seed:int = 1234, batch_size: int = 4, 
    epochs: int = 200
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    classes = (
        'tshirt-top', 'trouser', 'pullover', 'dress',
        'coat', 'sandal', 'shirt', 'sneaker', 
        'bag', 'ankle-boot'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root=ROOT_DIR, train=True,
        download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    testset = torchvision.datasets.FashionMNIST(
        root=ROOT_DIR, train=False,
        download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=True, num_workers=2
    )

    model = train_or_load_model(
        model=Net(), device=device,
        epochs=epochs, trainloader=trainloader,
        save_path=os.path.join(ROOT_DIR, 'fashion-mnist10.pth')
    )
    confusion_matrix = compute_confusion_matrix(
        model, dataloader=testloader,
        save_path=os.path.join(ROOT_DIR, 'fashion-mnist10_cfm.npy')
    )
    compute_treachorus_pairs(
        player_ids=classes, max_players=6, confusion_matrix=confusion_matrix,
        save_path=os.path.join(ROOT_DIR, 'players.yaml')
    )
   

if __name__ == '__main__':
    main()

