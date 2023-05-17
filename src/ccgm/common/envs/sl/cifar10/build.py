from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from ccgm.common.envs.sl.build import build
from ccgm.common.envs.sl.cifar10.config import ROOT_DIR
from ccgm.common.envs.sl.cifar10.net import Net



def main(
    seed: int = 1234, batch_size: int = 4, 
    epochs: int = 200, cfm_on_val: bool = False,
    max_players : int = 6
):
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    build(
        dataset_name="cifar10", dataset_fn=CIFAR10, classes=classes,
        transform=transform, root_dir=ROOT_DIR, net=Net(),
        seed=seed, batch_size=batch_size, epochs=epochs, cfm_on_val=cfm_on_val,
        max_players=max_players
    )
    

if __name__ == '__main__':
    main()
