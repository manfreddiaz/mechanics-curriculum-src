from collections import deque
from dataclasses import dataclass, field
import logging
from itertools import permutations
import math
import os
from typing import Any, List
import numpy as np

import pandas as pd
from sklearn import preprocessing

import hydra
from omegaconf import DictConfig

from ccgm.utils import CoalitionMetadata

import static.core as core
from static.utils import make_xpt_dir, hydra_custom_resolvers

log = logging.getLogger(__name__)


# hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="vpop")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    base_dir = os.path.join(
        cfg.run.outdir,
        f"{cfg.task.id}", 
    )
    indir = os.path.join(
        base_dir,
        f"{cfg.task.order}",
        f"{cfg.alg.id}"
    )
    
    assert os.path.exists(indir), "invalid step, run [main, eval, shapley] first"

    outdir = os.path.join(
        base_dir,
        "vpop",
        f"{cfg.task.order}",
        f"{cfg.alg.id}",
    )
    os.makedirs(outdir, exist_ok=True)


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
    vpop_dfs = []
    eval_teams = {team: None for team in meta_game.columns} # eval teams
    for team in eval_teams:
        scaler = preprocessing.MinMaxScaler((-1, 1))
        n_values = scaler.fit_transform(meta_game[team].to_numpy().reshape(-1, 1))
        meta_game[team].iloc[:, ] = n_values.flatten()
        vpop = core.vpop(meta_game[team], players)
        vpop_dfs.append(
            pd.DataFrame(vpop, index=players, columns=players)
        )
    
    vpop_df = pd.concat(vpop_dfs, keys=eval_teams.keys(), names=['eval_team'])
    vpop_df.to_pickle(
        os.path.join(outdir, 'vpop.pkl')
    )


if __name__ == '__main__':
    main()
