import csv
import logging
from itertools import permutations, product
import math
import os
from typing import List
import torch.multiprocessing as tmp
import numpy as np
from stable_baselines3.common.monitor import Monitor

import pandas as pd
from sklearn import preprocessing

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ccgm.utils import Coalition
from static.utils import make_xpt_dir, hydra_custom_resolvers

log = logging.getLogger(__name__)


def compute_shapley(values: pd.Series, players: list[str], ordered: bool = False) -> List[float]:
    scaler = preprocessing.MinMaxScaler((-1, 1))
    n_values = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
    values.iloc[:, ] = n_values.flatten()

    value = {player: 0.0 for player in players}
    players_idx = np.arange(len(players))
    players = np.array(players)

    def forward_dynamics(coalition: list):
        for player_idx in filter(lambda x: x not in coalition, players_idx):
            next_coalition = coalition + [player_idx]
            if not ordered:
                next_coalition_idx = sorted(next_coalition)
                coalition_idx = sorted(coalition)
            else:
                next_coalition_idx = next_coalition
                coalition_idx = coalition            
            # maintain permutation invariance by order
            next_coalition_id = Coalition.to_id(players[next_coalition_idx])
            # print(next_coalition, coalition)
            if len(coalition) > 0:
                coalition_id = Coalition.to_id(players[coalition_idx])
                value[players[player_idx]] += values[next_coalition_id] - values[coalition_id]
            else:
                value[players[player_idx]] += values[next_coalition_id]
            
            forward_dynamics(next_coalition)
    
    forward_dynamics([])
    
    norm = math.factorial(players.shape[0])
    value = {player: value * norm ** -1 for player, value in value.items()}

    return value


def compute_nowak_radzik(values: pd.Series, players: list[str]) -> List[float]:
    return compute_shapley(values, players, True)

def compute_sanchez_bergantinos(values: pd.Series, players: list[str]) -> List[float]:
    scaler = preprocessing.MinMaxScaler((-1, 1))
    n_values = scaler.fit_transform(values.to_numpy().reshape(-1, 1))
    values.iloc[:, ] = n_values.flatten()

    value = {player: 0.0 for player in players}
    players_idx = np.arange(len(players))
    players = np.array(players)

    def backward_dynamics(coalition: list):
        for player in coalition:
            next_coalition = list(coalition)
            next_coalition.remove(player)
            coalition_id = Coalition.to_id(players[coalition])
            # log.info((coalition, next_coalition))
            if len(next_coalition) > 0:
                next_coalition_id = Coalition.to_id(players[next_coalition])
                value[players[player]] += values[coalition_id] - values[next_coalition_id]
            else:
                value[players[player]] += values[coalition_id]
            
            backward_dynamics(next_coalition)


    for coalition in permutations(players_idx):
        backward_dynamics(list(coalition))

    norm = math.factorial(players.shape[0]) ** 2
    value = {player: value * norm ** -1 for player, value in value.items()}

    return value


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="shapley")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    indir = make_xpt_dir(cfg)

    df = pd.read_csv(os.path.join(indir, 'results.csv'))
    df = df.groupby(['train_team', 'eval_team'])

    # compute initial and final performance across traini seeds
    # and evaluation seeds
    initial_perf = df["r_0"].agg('mean')
    final_perf = df['r_1'].agg('mean')

    if cfg.metric == 'contrib':
        # contrib: performance at initialization vs final performance
        metric = final_perf - initial_perf
    elif cfg.metric == 'final':
        # final: final performance, treat initialization as 0
        metric = final_perf
    else:
        raise NotImplementedError()
    
    if cfg.method == 'shapley':
        assert cfg.task.order == 'random'            
        method = compute_shapley
    elif cfg.method == 'nowak_radzik':
        assert cfg.task.order == 'ordered'
        method = compute_nowak_radzik
    elif cfg.method == 'sanchez_bergantinos':
        assert cfg.task.order == 'ordered'
        method = compute_sanchez_bergantinos
    else:
        raise NotImplementedError(cfg.method)
    
    meta_game = metric.reset_index()
    meta_game = meta_game.pivot_table(
        values=meta_game.columns[-1], 
        index='train_team', 
        columns='eval_team'
    )
    meta_game.to_csv(os.path.join(indir, 'meta_game.csv'))

    task, _ = hydra.utils.instantiate(cfg.task)
    players = [player for player in task.players]
    # trainer cooperative game
    eval_teams = {team: None for team in meta_game.columns} # eval teams
    for team in eval_teams:
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
        evaluator_df.to_csv(os.path.join(indir, f'evaluator_{cfg.method}_{cfg.metric}.csv'))


if __name__ == '__main__':
    main()
