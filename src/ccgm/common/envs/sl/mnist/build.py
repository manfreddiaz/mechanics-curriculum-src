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

from ccgm.common.envs.sl.mnist.config import ROOT_DIR
from ccgm.common.envs.sl.mnist.net import Net
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
        'digit0', 'digit1', 'digit2', 'digit3',
        'digit4', 'digit5', 'digit6', 'digit7', 
        'digit8', 'digit9'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root=ROOT_DIR, train=True,
        download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=2
    )

    model = train_or_load_model(
        model=Net(), device=device,
        epochs=epochs, trainloader=trainloader,
        save_path=os.path.join(ROOT_DIR, 'mnist10.pth')
    )
    confusion_matrix = compute_confusion_matrix(
        model, dataloader=trainloader,
        save_path=os.path.join(ROOT_DIR, 'mnist10_cfm.npy')
    )
    compute_treachorus_pairs(
        player_ids=classes, max_players=6, confusion_matrix=confusion_matrix,
        save_path=os.path.join(ROOT_DIR, 'players.yaml')
    )
   

if __name__ == '__main__':
    main()

