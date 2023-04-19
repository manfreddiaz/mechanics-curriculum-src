import pickle
import torch.utils.data as data

from ccgm.common.envs.sl.cifar10.config import ROOT_DIR

class PlayerDataset(data.Dataset):
    
    def __init__(
        self,
        player_id: int
    ) -> None:
        super().__init__()
    
        self.data = []
        self.targets = []

        self.player_file = os.path.join(ROOT_DIR, f"player_{player_id}.pkl")
        with open(self.player_file, mode='rb') as f:
            try:
                while True:
                    image, label = pickle.load(f)
                    self.data.append(image)
                    self.targets.append(label)
            except EOFError:
                pass

    def __getitem__(self, idx: int):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import os
    ds = PlayerDataset(
        player_id=0
    )
    ds






