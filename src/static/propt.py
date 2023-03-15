import csv
import json
import logging
import os
import traceback
import torch.multiprocessing as tmp
import torch
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig

from ccgm.utils import CoalitionMetadata

from static.utils import hydra_custom_resolvers
from static.utils import play

def proportional_shapley(values: np.array):
    propt = values - values.max()
    propt += 1e-5
    propt = np.exp(propt)
    propt /= np.sum(propt)

    return propt


def coalition_from_shapley_value(players: list[str], indir: str, cfg: DictConfig):
    value = pd.read_csv(
        os.path.join(indir, f"trainer_{cfg.propt.method}_final.csv"),
        index_col=0
    )
    # NOTE: Shapley values are additive
    propt = proportional_shapley(value.sum(axis=1)).to_numpy()
    if cfg.task.order == "ordered":
        sorted_idx = propt.argsort()[::-1]
        coalition = CoalitionMetadata(
            players=players[sorted_idx].tolist(),
            idx=0,
            ordered=True,
            probs=propt[sorted_idx].tolist()
        )
    elif cfg.task.order == "random":
        coalition = CoalitionMetadata(
            players=players.tolist(),
            idx=0,
            ordered=False,
            probs=propt.tolist()
        )
    else:
        raise ValueError(f"invalid value for <task.order>: {cfg.task.order}")

    return coalition


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="propt")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    torch.backends.cudnn.deterministic = cfg.torch.deterministic
    
    game_spec, _ = hydra.utils.instantiate(cfg.task)

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
        "propt",
        f"{cfg.task.order}",
        f"{cfg.alg.id}",
        f"{cfg.propt.method}",
    )
    os.makedirs(outdir, exist_ok=True)

    seeds = range(cfg.run.seed, cfg.run.seed + cfg.run.num_seeds)
    games = {}

    coalition = coalition_from_shapley_value(
        np.array([player_id for player_id in game_spec.players]),
        indir=indir,
        cfg=cfg
    )

    with open(os.path.join(outdir, 'game.json'), mode='w+') as f:
        json.dump(vars(coalition), f, indent=2)

    tmp.set_start_method('spawn')
    with tmp.Pool(
        processes=cfg.thread_pool.size, 
        maxtasksperchild=cfg.thread_pool.maxtasks
    ) as ppe:
        for seed in seeds:
            log.info(f"<submit> game with {coalition.id}, seed: {seed}")
            games[ppe.apply_async(play, (coalition, seed, outdir, cfg))] = coalition

        for game in games:
            try:
                code = game.get()
                coalition = games[game]
                log.info(f'<finished> game with {coalition.id}: finished with exit code {code}')
            except Exception as ex:
                log.error(
                    f'<FAIL> game {coalition.id} with the following exception: \n',
                    traceback.format_exc(),
                    exc_info=ex
                )


if __name__ == '__main__':
    main()
