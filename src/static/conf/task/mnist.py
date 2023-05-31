
from torch.utils.data import DataLoader
from ccgm.common.envs.sl.env import ClassCoalitionDataset
from ccgm.common.envs.sl.mnist.config import ROOT_DIR
from ccgm.utils import CoalitionMetadata, CoalitionalGame
from ccgm.common.envs.sl.mnist import (
    SPURIOUS
)


def make_cooperative_task(
    team: list[int], train: bool,
    batch_size: int,
    num_workers: int
):
    coalition_dataset = ClassCoalitionDataset(
        player_ids=team, root_dir=ROOT_DIR,
        train=train
    )
    return DataLoader(
        coalition_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )


def make_task(
    id: str,
    order: str,
    version: str,
    batch_size: int,
    num_workers: int,
    train: bool
):
    if version == 'spur':
        players = SPURIOUS
    else:
        raise ValueError(version)
    
    if order == 'ordered':
        # ChainedDataset
        raise ValueError(order)

    game_spec = CoalitionalGame.make(
        players=list(map(str, players)),
        ordered=False    
    )

    def make_game(
        team: CoalitionMetadata, probs: list[int], 
        team_dir: str, seed: int, train: bool = train
    ):
        if probs is not None:
            raise ValueError("probs")

        return make_cooperative_task(
            team=list(map(int, team.players)), train=train,
            batch_size=batch_size, num_workers=num_workers
        )

    return game_spec, make_game
