import itertools
import logging
import os
import hydra
import networkx as nx
from networkx.algorithms.components import attracting_components, strongly_connected_components
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from ccgm.utils import CoalitionMetadata
import matplotlib.pyplot as plt


import static.csc as csc
from static.utils import hydra_custom_resolvers, make_xpt_dir

log = logging.getLogger(__name__)


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="stability")
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
        evaluators = [player for player in task.players]
    elif cfg.setting == 'all':
        evaluators = [CoalitionMetadata.to_id(task.players)]
    else:
        raise ValueError(f'cfg.setting == {cfg.setting}')

    players_idx = np.arange(len(task.players))
    players = np.array([player for player in task.players])
    for i, evaluator in enumerate(evaluators):
        meta_game[evaluator].iloc[:, ] = meta_game[evaluator].to_numpy().flatten()

        graph = nx.DiGraph()
        graph.add_node('empty', value=0.0)

        for coalition in meta_game[evaluator].index:
            graph.add_node(coalition, value=meta_game[evaluator].loc[coalition])

        for permutation in itertools.permutations(players_idx):
            for idx, player in enumerate(permutation):
                if idx == 0:
                    coalition_id = 'empty'
                else:
                    coalition = sorted(permutation[:idx])
                    coalition_id = CoalitionMetadata.to_id(players[coalition])
                
                next_coalition = sorted(permutation[: idx+1])
                next_coalition_id = CoalitionMetadata.to_id(players[next_coalition])
            
                if coalition_id == 'empty':
                    marginal = meta_game[evaluator].loc[next_coalition_id]  # TODO: missing value for initialization
                else:  
                    marginal = meta_game[evaluator].loc[next_coalition_id] - meta_game[evaluator].loc[coalition_id]

                if marginal >= 0:
                    graph.add_edge(coalition_id, next_coalition_id, value=marginal)
                else:
                    graph.add_edge(next_coalition_id, coalition_id, value=marginal)
        figure = plt.figure(figsize=(16,9))        
        # nx.drawing.layout.(graph)
        nx.draw(
            graph,
            pos=nx.layout.fruchterman_reingold_layout(graph),
            with_labels=True
        )
        plt.suptitle(f'{evaluator}')
        plt.savefig(f'{i}.pdf')
        print(f"evaluator: {evaluator}")
        for idx, ac in enumerate(attracting_components(graph)):
            print("\t", idx, ac)
        



if __name__ == '__main__':
    main()
