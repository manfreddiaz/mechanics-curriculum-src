import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from ccgm.common.envs.sl.mnist.config import ROOT_DIR
from ccgm.common.envs.sl.gamify import gamify


def main():
    for train in [True, False]:
        gamify(
            root_dir=ROOT_DIR, dataset_fn=MNIST,
            train=train, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

if __name__ == '__main__':
    main()
