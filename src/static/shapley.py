import logging
from itertools import permutations
import math
import os
from typing import List
import numpy as np

import pandas as pd
from sklearn import preprocessing

import hydra
from omegaconf import DictConfig

import static.csc as csc
from static.utils import make_xpt_dir, hydra_custom_resolvers

log = logging.getLogger(__name__)


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="shapley")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    indir = make_xpt_dir(cfg)

    meta_game = pd.read_csv(
        os.path.join(indir, 'meta_game.csv'),
        index_col=0
    )

    if cfg.method == 'shapley':
        assert cfg.task.order == 'random'            
        method = csc.functional.shapley
    elif cfg.method == 'nowak_radzik':
        assert cfg.task.order == 'ordered'
        method = csc.functional.nowak_radzik
    elif cfg.method == 'sanchez_bergantinos':
        assert cfg.task.order == 'ordered'
        method = csc.functional.sanchez_bergantinos
    else:
        raise NotImplementedError(cfg.method)

    task, _ = hydra.utils.instantiate(cfg.task)
    players = [player for player in task.players]
    # trainer cooperative game

    eval_teams = {team: None for team in meta_game.columns} # eval teams
    for team in eval_teams:
        # scaler = preprocessing.MinMaxScaler((-1, 1))
        # n_values = scaler.fit_transform(meta_game[team].to_numpy().reshape(-1, 1))
        meta_game[team].iloc[:, ] = meta_game[team].to_numpy().flatten()
        eval_teams[team] = method(meta_game[team], players)
    # save trainers
    trainer_df = pd.DataFrame.from_dict(eval_teams)
    trainer_df.to_csv(os.path.join(indir, f'trainer_{cfg.method}_{cfg.metric}.csv'))

    if cfg.task.order == 'random': 
        # evaluator cooperative game
        train_teams = {team: None for team in meta_game.index}
        for team in train_teams:
            train_teams[team] = method(meta_game.loc[team] * -1.0, players)
    
        evaluator_df = pd.DataFrame.from_dict(train_teams)
        evaluator_df.to_csv(
            os.path.join(indir, f'evaluator_{cfg.method}_{cfg.metric}.csv')
        )


if __name__ == '__main__':
    main()
