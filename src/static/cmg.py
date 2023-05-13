import logging
import os
import pandas as pd
import hydra
from omegaconf import DictConfig

from static.utils import make_xpt_dir, hydra_custom_resolvers


log = logging.getLogger(__name__)

hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="shapley")
def main(
    cfg: DictConfig
) -> None:
    """
     Computes the cooperative meta-game resulting from the cooperative training
     and evaluation over all coalitions.
    """
    
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
    
    meta_game = metric.reset_index()
    meta_game = meta_game.pivot_table(
        values=meta_game.columns[-1], 
        index='train_team', 
        columns='eval_team'
    )
    meta_game.to_csv(os.path.join(indir, 'meta_game.csv'))

    log.info("cooperative meta game computed.")


if __name__ == "__main__":
    main()
