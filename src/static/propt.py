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
import static.proj as proj

from static.utils import hydra_custom_resolvers
from static.utils import play


def coalition_from_shapley_value(
    players: list[str], 
    indir: str, 
    cfg: DictConfig,
):
    if cfg.propt.method in ["shapley", "nowak_radzik", "sanchez_bergantinos"]:
        value = pd.read_csv(
            os.path.join(indir, f"trainer_{cfg.propt.method}_final.csv"),
            index_col=0
        )
        # NOTE: Shapley values are additive
        if cfg.propt.reduce == "mean":
            priors = value.mean(axis=1).to_numpy()
        elif cfg.propt.reduce == "sum":
            priors = value.sum(axis=1).to_numpy()
        elif cfg.propt.reduce == "all":
            priors = value[
                CoalitionMetadata.to_id(players)].to_numpy()
        elif cfg.propt.reduce == "player":
            assert "player" in cfg.propt
            priors = value[
                cfg.propt.player].to_numpy()

    elif cfg.propt.method == "uniform":
        priors = np.ones(shape=(len(players)))
    
    if cfg.propt.proj == "softmax":
        propt = proj.projection_softmax(priors)
    elif cfg.propt.proj == "simplex":
        propt = proj.projection_simplex_sort(priors)
    else:
        raise ValueError("propt.proj")
    
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
        raise ValueError("task.order")

    return coalition, priors


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

    assert os.path.exists(
        indir), "invalid step, run [main, eval, shapley] first"

    outdir = os.path.join(
        base_dir,
        "propt",
        f"{cfg.task.order}",
        f"{cfg.alg.id}",
        f"{cfg.propt.method}-{cfg.propt.reduce}-{cfg.propt.proj}",

    )
    os.makedirs(outdir, exist_ok=True)

    seeds = range(cfg.run.seed, cfg.run.seed + cfg.run.num_seeds)
    games = {}

    players = np.array([player_id for player_id in game_spec.players])
    coalition, priors = coalition_from_shapley_value(
        players,
        indir=indir,
        cfg=cfg
    )
    eval_coalition = CoalitionMetadata(
        players=players.tolist(),
        idx=0,
        ordered=False,
        probs=proj.projection_softmax(
            np.ones(shape=len(players)
        )).tolist()
    )

    info = vars(coalition)
    info['prior'] = priors.tolist()
    with open(os.path.join(outdir, 'game.json'), mode='w+') as f:
        json.dump(info, f, indent=2)

    
    tmp.set_start_method('spawn')
    with tmp.Pool(
        processes=cfg.thread_pool.size,
        maxtasksperchild=cfg.thread_pool.maxtasks
    ) as ppe:
        for seed in seeds:
            log.info(f"<submit> game with {coalition.id}, seed: {seed}")
            games[ppe.apply_async(
                play, (coalition, eval_coalition, seed, outdir, cfg))] = coalition

        for game in games:
            try:
                code = game.get()
                coalition = games[game]
                log.info(
                    f'<finished> game with {coalition.id}:' +
                    f'finished with exit code {code}')
            except Exception as ex:
                log.error(
                    f'<FAIL> game {coalition.id} with the following exception: \n',
                    traceback.format_exc(),
                    exc_info=ex
                )


if __name__ == '__main__':
    main()
