import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms

from ccgm.common.envs.sl.mnist.config import ROOT_DIR


def main():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(
        ROOT_DIR, train=True, download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4,
        shuffle=True, num_workers=2)

    players_file = [
        open(f'{ROOT_DIR}/player_{idx}.pkl', mode='wb+') for idx in range(10)]
    for data in trainloader:
        images, labels = data
        for image, label in zip(images, labels):
            pickle.dump((image, label), players_file[label])
    


if __name__ == '__main__':
    main()
