import logging
import os
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from ccgm.utils import CoalitionMetadata

import static.csc as csc
from static.utils import hydra_custom_resolvers, make_xpt_dir

log = logging.getLogger(__name__)


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="rational")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    indir = make_xpt_dir(cfg)

    meta_game = pd.read_csv(
        os.path.join(indir, 'meta_game.csv'),
        index_col=0
    )

    task, _ = hydra.utils.instantiate(cfg.task)
    
    
    if cfg.setting == "target":
        players = [player for player in task.players]
    elif cfg.setting == 'all':
        players = [CoalitionMetadata.to_id(task.players)]
    else:
        raise ValueError(f'cfg.setting == {cfg.setting}')

    entries = []
    for player in players:
        meta_game[player].iloc[:, ] = meta_game[player].to_numpy().flatten()
        # compute the Shapley allocations on all subgames, including the empty
        # subgame_shapley = csc.functional.subgames_shapley(meta_game[player], players)
        # subgame_shapley.pop('') # remove empty subgame
        # subgames = list(subgame_shapley.keys())
        # convert the subgames into an array of 2^n - 1 x n 
        coalitions_array = meta_game[player].to_numpy()
        # subgame_shapley_array = subgame_shapley_array.sum(axis=-1)
        # the maximum gain in any subgame allocation for player with index idx
        # is in the column of the subgames on idx
        max_coalitional_idx = np.argmax(coalitions_array)
        # v_{ii} - max_{S \in G^S} \phi(i, v^S, S)
        rationality = meta_game[player].loc[player] - coalitions_array[max_coalitional_idx]
        # a curriculum game is rational if the player gains
        # from cooperation
        is_rational = rationality < 0.0

        max_ids = np.flatnonzero([coalitions_array == coalitions_array.max()])
        log.info(f'processing: {player}')

        for max_id in max_ids:
            entries.append({
                "player": player, 
                "is_rational": is_rational, 
                "rational_value": rationality, 
                "coalition": meta_game[player].index[max_id], 
                "coalition_value": coalitions_array[max_coalitional_idx]
            })

    df = pd.DataFrame.from_records(entries)
    df.to_csv(os.path.join(indir, "rational.csv" if cfg.setting == 'target' else "rational_all.csv"))



if __name__ == '__main__':
    main()
