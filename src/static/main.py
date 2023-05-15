import logging
import os
from itertools import product
import traceback
import torch.multiprocessing as tmp
import torch

import hydra
from omegaconf import DictConfig

from static.utils import hydra_custom_resolvers, play


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    torch.backends.cudnn.deterministic = cfg.torch.deterministic
    
    game_spec, _ = hydra.utils.instantiate(cfg.task)

    outdir = os.path.join(
        cfg.run.outdir,
        f"{cfg.task.id}", 
        f"{cfg.task.order}",
        f"{cfg.alg.id}"
    )
    os.makedirs(outdir, exist_ok=True)

    seeds = range(cfg.run.seed, cfg.run.seed + cfg.run.num_seeds)
    games = {}

    tmp.set_start_method('spawn')
    with tmp.Pool(
        processes=cfg.thread_pool.size, 
        maxtasksperchild=cfg.thread_pool.maxtasks
    ) as ppe:
        for seed, coalition in product(seeds, game_spec.coalitions):
            log.info(f"<submit> game with {coalition.id}, seed: {seed}")
            games[ppe.apply_async(play, (coalition, None, seed, outdir, cfg))] = coalition

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
