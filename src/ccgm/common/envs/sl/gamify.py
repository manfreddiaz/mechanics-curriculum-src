import os
import pickle
import torch

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def gamify(
    root_dir: str, dataset_fn: VisionDataset,
    transform: transforms.Compose, train: bool = False
):
    dataset = dataset_fn(
        root=root_dir, train=train,
        download=True, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=4,
        shuffle=True, num_workers=2)

    base_dir = os.path.join(root_dir, "train" if train else "test")
    os.makedirs(base_dir, exist_ok=True)

    players_file = [
        open(f'{base_dir}/player_{idx}.pkl', mode='wb+') 
            for idx in range(10)]
    for data in loader:
        images, labels = data
        for image, label in zip(images, labels):
            pickle.dump((image, label), players_file[label])
