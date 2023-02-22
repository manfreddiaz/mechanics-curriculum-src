import logging
import os
import random
from itertools import product
import traceback
import torch.multiprocessing as tmp
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor

import hydra
from omegaconf import DictConfig, OmegaConf

from ccgm.utils import Coalition


def play(
    coalition: Coalition, 
    seed, 
    outdir, 
    cfg: DictConfig
):
    log = logging.getLogger()
    # loggging and saving config
    team_dir = os.path.join(outdir, f'game-{str(coalition.idx)}')
    os.makedirs(team_dir, exist_ok=True)
   
    # avoid duplicated runs on restart
    final_model = os.path.join(team_dir, f'{seed}f.model.ckpt')
    if os.path.exists(final_model):
        log.info(f"<duplicate> game {coalition.id} with: {coalition.id}, seed: {seed}")
        return 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log.info(f"<playing> game {coalition.idx} with: {coalition.id}, seed: {seed}")
    
    log.info(f'<build> environment {cfg.task.id} from config.')
    _, game_factory = hydra.utils.instantiate(cfg.task)
    make_env = game_factory(coalition)
    def monitored():
        return Monitor(
            make_env(),
            filename=os.path.join(team_dir, f'{seed}.train'),
            info_keywords=('meta-strategy',)
        )
    envs = gym.vector.SyncVectorEnv([
        monitored for i in range(cfg.task.num_envs)
    ])
    envs.seed(seed)

    log.info(f'<build> agent {cfg.agent.id} from config.') 
    make_agent = hydra.utils.instantiate(cfg.agent)
    agent = make_agent(envs)
    torch.save(agent, os.path.join(team_dir, f'{seed}i.model.ckpt'))

    log.info(f'<build> algorithm {cfg.alg.id} from config')
    make_alg = hydra.utils.instantiate(cfg.alg)
    algorithm = make_alg(envs=envs)
    
    log.info(f'<learn>')
    algorithm.learn(
        agent=agent,
        task=cfg.task,
        logger=SummaryWriter(os.path.join(team_dir, f'tb-{str(seed)}')),
        device=torch.device(cfg.torch.device),
    )

    torch.save(
        agent, 
        final_model
    )

    game_info_file = os.path.join(team_dir, 'game.info')
    # log game info
    if not os.path.exists(game_info_file):
        with open(game_info_file, mode='w') as f:
            f.write(coalition.id)
            f.write('\r\n')
            f.write(cfg.alg.id)

    log.info(f"<completed> game with: {coalition.id}, seed: {seed}")
    return 0

OmegaConf.register_new_resolver(
    "bmult", lambda x, y: x * y
)

OmegaConf.register_new_resolver(
    "bdiv", lambda x, y: x // y
)



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
