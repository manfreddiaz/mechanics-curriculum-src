import logging
import os
import random
from itertools import product
import traceback
import torch.multiprocessing as tmp
import gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

import hydra
from omegaconf import DictConfig, OmegaConf

from ccgm.utils import Coalition

log = logging.getLogger(__name__)


def make_input_dir(
    training_coalition: Coalition,
    cfg: DictConfig
):
    indir = os.path.join(
        cfg.run.outdir,
        f"{cfg.task.id}", 
        f"{cfg.task.order}",
        f"{cfg.alg.id}"
    )
    return os.path.join(
        indir, 
        f'game-{str(training_coalition.idx)}'
    )


def load_coalition_models(
    training_coalition: Coalition,
    seed: int,
    cfg: DictConfig
):
    team_dir = make_input_dir(training_coalition, cfg)
    
    # verify if the initial model was produced
    # fail otherwise
    initial_model = os.path.join(team_dir, f'{seed}i.model.ckpt')
    if not os.path.exists(initial_model):
        log.error(f"<failed> game {training_coalition.id} may have never started.")
        raise FileNotFoundError(initial_model)
    
    # verify if the final model was produced
    # fail otherwise
    final_model = os.path.join(team_dir, f'{seed}f.model.ckpt')
    if not os.path.exists(final_model):
        log.error(f"<failed> game {training_coalition.id} may have not completed!")
        raise FileNotFoundError(final_model)
    
    i_model = torch.load(initial_model)
    f_model = torch.load(final_model)

    return i_model, f_model


def eval(
    training_coalition: Coalition,
    evaluation_coalition: Coalition, 
    train_seed: int,
    eval_seed: int,
    cfg: DictConfig
):
 
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    train_dir = make_input_dir(training_coalition, cfg)
    init_agent, final_agent = load_coalition_models(
        training_coalition=training_coalition,
        seed=train_seed,
        cfg=cfg
    )
    
    _, game_factory = hydra.utils.instantiate(cfg.eval.task)
    make_env = game_factory(evaluation_coalition)
    
    def monitored():
        return Monitor(
            make_env(),
            filename=os.path.join(train_dir, f'{train_seed}.eval'),
            info_keywords=('meta-strategy',)
        )

    envs = gym.vector.SyncVectorEnv([
        monitored()
    ])

    game_info_file = os.path.join(train_dir, 'eval.info')
    # log game info
    with open(game_info_file, mode='w+') as f:
        f.write(training_coalition.id)
        f.write('\r\n')
        f.write(cfg.alg.id)

    log.info(f"<completed> game with: {training_coalition.id}, seed: {eval_seed}")
    return 0

OmegaConf.register_new_resolver(
    "bmult", lambda x, y: x * y
)

OmegaConf.register_new_resolver(
    "bdiv", lambda x, y: x // y
)

def hydra_load_node(x: str):
    cfg = hydra.compose(f"{x}.yaml")
    cfg = cfg[list(cfg.keys())[0]]
    return cfg

OmegaConf.register_new_resolver(
    "load", hydra_load_node
)

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    torch.backends.cudnn.deterministic = cfg.torch.deterministic
    
    train_game_spec, _ = hydra.utils.instantiate(cfg.task)
    eval_game_spec, _ = hydra.utils.instantiate(cfg.eval.task)

    training_seeds = range(cfg.run.seed, cfg.run.seed + cfg.run.num_seeds)
    evaluation_seeds = range(cfg.eval.seed, cfg.eval.seed + cfg.eval.num_seeds)

    games = {}

    tmp.set_start_method('spawn')
    with tmp.Pool(
        processes=cfg.thread_pool.size, 
        maxtasksperchild=cfg.thread_pool.maxtasks
    ) as ppe:
        for train_seed, eval_seed, training_coalition, evaluation_coalition in product(
            training_seeds, evaluation_seeds, train_game_spec.coalitions, eval_game_spec.coalitions
        ):
            log.info(
                f"<submit> eval game {training_coalition.id}-{train_seed} vs {evaluation_coalition.id}-{eval_seed}" 
            )
            async_result = ppe.apply_async(
                eval, (training_coalition, evaluation_coalition, train_seed, eval_seed, cfg)
            )
            games[async_result] = (training_coalition, evaluation_coalition)

        for game in games:
            try:
                code = game.get()
                training_coalition = games[game]
                log.info(f'<finished> game with {training_coalition.id}: finished with exit code {code}')
            except Exception as ex:
                log.error(
                    f'<FAIL> game {training_coalition.id} with the following exception: \n',
                    traceback.format_exc(),
                    exc_info=ex
                )


if __name__ == '__main__':
    main()
