import logging
import os


import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import static.csc as core
from static.utils import hydra_custom_resolvers, make_xpt_dir

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

    if cfg.solution_concept == "sanchez_bergantinos":
        assert cfg.task.order == "ordered"
        solution_concept = core.functional.sanchez_bergantinos
    elif cfg.solution_concept == "nowak_radzik":
        assert cfg.task.order == "ordered"
        solution_concept = core.functional.nowak_radzik
    elif cfg.solution_concept == "shapley":
        assert cfg.task.order == "random"
        solution_concept = core.functional.shapley

    assert os.path.exists(indir), "invalid step, run [main, eval, cmg] first"

    outdir = os.path.join(
        base_dir,
        "vpop",
        f"{cfg.task.order}",
        f"{cfg.alg.id}",
        f"{cfg.solution_concept}"
    )
    os.makedirs(outdir, exist_ok=True)

    meta_game = pd.read_csv(
        os.path.join(indir, 'meta_game.csv'),
        index_col=0
    )

    task, _ = hydra.utils.instantiate(cfg.task)
    players = [player for player in task.players]
    # trainer cooperative game
    vpop_dfs = []
    eval_teams = [team for team in meta_game.columns] # eval teams
    for team in eval_teams:
        meta_game[team].iloc[:, ] = meta_game[team].to_numpy().flatten()
        vpop = core.functional.vpop(
            meta_game[team], players,
            solution_concept=solution_concept, ordered=cfg.task.order
        )
        vpop_dfs.append(
            pd.DataFrame(vpop, index=players, columns=players)
        )

    vpop_df = pd.concat(vpop_dfs, keys=eval_teams, names=['eval_team'])
    vpop_df.to_pickle(
        os.path.join(outdir, f'vpop.pkl')
    )


if __name__ == '__main__':
    main()
