from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from ccgm.common.envs.sl.cifar10.config import ROOT_DIR
from ccgm.common.envs.sl.gamify import gamify


def main():
    for train in [True, False]:
        gamify(
            root_dir=ROOT_DIR, dataset_fn=CIFAR10,
            train=train, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )


if __name__ == '__main__':
    main()
