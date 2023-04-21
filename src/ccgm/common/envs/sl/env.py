import os
import pickle
import torch.utils.data as data


class ClassCoalitionDataset(data.Dataset):
    
    def __init__(
        self,
        player_ids: list[int],
        root_dir: str
    ) -> None:
        super().__init__()
    
        self.data = []
        self.targets = []

        for player_id in player_ids:
            self.player_file = os.path.join(root_dir, f"player_{player_id}.pkl")
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
    ds = ClassCoalitionDataset(
        player_id=0
    )
    ds