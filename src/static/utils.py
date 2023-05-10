
import functools
import logging
import os
import random
from typing import Callable
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from ccgm.common.coalitions import Coalition


def make_xpt_dir(cfg):
    return os.path.join(
        cfg.run.outdir,
        f"{cfg.task.id}", 
        f"{cfg.task.order}",
        f"{cfg.alg.id}"
    )

def make_xpt_coalition_dir(
    training_coalition: Coalition,
    cfg: DictConfig
):
    
    return os.path.join(
        make_xpt_dir(cfg), 
        f'game-{str(training_coalition.idx)}'
    )

def _hydra_load_node(x: str):
    cfg = hydra.compose(f"{x}.yaml")
    cfg = cfg[list(cfg.keys())[0]]
    return cfg

def hydra_custom_resolvers():
    OmegaConf.register_new_resolver(
        "bmult", lambda x, y: x * y
    )

    OmegaConf.register_new_resolver(
        "bdiv", lambda x, y: x // y
    )

    OmegaConf.register_new_resolver(
        "load", _hydra_load_node
    )


def play(
    train_coalition: Coalition, 
    eval_coalition: Coalition,
    seed, 
    outdir, 
    cfg: DictConfig,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if GlobalHydra.instance() is not None:
        if not GlobalHydra.instance().is_initialized():
            hydra.initialize(version_base=None, config_path='conf')

    log = logging.getLogger()
    # loggging and saving config
    team_dir = os.path.join(outdir, f'game-{str(train_coalition.idx)}')
    os.makedirs(team_dir, exist_ok=True)
   
    # avoid duplicated runs on restart
    final_model = os.path.join(team_dir, f'{seed}f.model.ckpt')
    if os.path.exists(final_model):
        log.info(f"<duplicate> game {train_coalition.id} with: {train_coalition.id}, seed: {seed}")
        return 0

    log.info(f"<playing> game {train_coalition.idx} with: {train_coalition.id}, seed: {seed}")
    
    # Environment
    log.info(f'<build> environment {cfg.task.id} from config.')
    _, make_env = hydra.utils.instantiate(cfg.task)
    env = make_env(
        train_coalition, train_coalition.probs,
        team_dir, seed
    )

    eval_fn = None
    # Evaluation
    if cfg.eval:
        log.info(f'<build> eval environment {cfg.task.id} from config.')
        _, make_env = hydra.utils.instantiate(cfg.eval.task)
        eval_env = make_env(
            eval_coalition, eval_coalition.probs,
            team_dir, seed, train=False
        )
        eval_fn = hydra.utils.instantiate(cfg.eval.evaluator)
        eval_fn = functools.partial(
            eval_fn,
            env=eval_env,
            num_steps=cfg.eval.total_timesteps,
            device=torch.device(cfg.torch.device)
        )


    # Agent
    log.info(f'<build> agent {cfg.agent.id} from config.') 
    make_agent = hydra.utils.instantiate(cfg.agent)
    agent = make_agent(
        envs=env,
        hparams=cfg.alg.hparams,
        device=torch.device(cfg.torch.device)
    )
    torch.save(
        agent, 
        os.path.join(team_dir, f'{seed}i.model.ckpt')
    )

    # Learning Algorithm
    log.info(f'<build> algorithm {cfg.alg.id} from config')
    make_alg = hydra.utils.instantiate(cfg.alg)
    learn_fn = make_alg(
        envs=env
    )
    
    log.info(f'<learn>')
    learn_fn(
        agent=agent,
        logger=SummaryWriter(os.path.join(team_dir, f'tb-{str(seed)}')),
        device=torch.device(cfg.torch.device),
        log_every=cfg.run.log_every,
        log_file_format=os.path.join(team_dir, f'{seed}' + '-{}.model.ckpt'),
        eval_fn = eval_fn
    )

    torch.save(
        agent, 
        final_model
    )

    game_info_file = os.path.join(team_dir, 'game.info')
    # log game info
    if not os.path.exists(game_info_file):
        with open(game_info_file, mode='w') as f:
            f.write(train_coalition.id)
            f.write('\r\n')
            f.write(cfg.alg.id)

    log.info(f"<completed> game with: {train_coalition.id}, seed: {seed}")
    return 0



