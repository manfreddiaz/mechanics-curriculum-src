import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from ccgm.common.envs.sl.build import build
from ccgm.common.envs.sl.mnist.config import ROOT_DIR
from ccgm.common.envs.sl.mnist.net import Net


def main(
    seed:int = 1234, batch_size: int = 4, 
    epochs: int = 200, cfm_on_val: bool = False,
    max_players: int = 6
):

    classes = (
        'digit0', 'digit1', 'digit2', 'digit3',
        'digit4', 'digit5', 'digit6', 'digit7', 
        'digit8', 'digit9'
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    build(
        dataset_name="mnist", dataset_fn=MNIST, classes=classes,
        transform=transform, root_dir=ROOT_DIR, net=Net(),
        seed=seed, batch_size=batch_size, epochs=epochs, cfm_on_val=cfm_on_val,
        max_players=max_players
    )

if __name__ == '__main__':
    main()

