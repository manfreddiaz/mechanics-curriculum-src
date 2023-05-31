from distutils.util import strtobool
import logging
import os
import random
import traceback
import numpy as np

import gym
import hydra
from omegaconf import DictConfig
import torch

from stable_baselines3.common.monitor import Monitor
from SMPyBandits.Policies import Exp3S, Hedge

from learned.utils import hydra_custom_resolvers
from tensorboardX import SummaryWriter
import torch.multiprocessing as tmp


hydra_custom_resolvers()


def make_meta_agent(cfg: DictConfig, envs):
    make_meta_agent = hydra.utils.instantiate(cfg.meta.agent)
    meta_agent = make_meta_agent(
        envs=envs,
        hparams=cfg.meta.alg.hparams,
        device=torch.device(cfg.torch.device)
    )

    make_meta_alg = hydra.utils.instantiate(cfg.meta.alg)
    meta_alg_play_fn, meta_alg_optim_fn, _, _ = make_meta_alg(
        logger=None,
        device=torch.device(cfg.torch.device),
        log_every=cfg.run.log_every,
        log_file_format=None
    )

    return meta_agent, meta_alg_play_fn, meta_alg_optim_fn


def play_bandit(seed, outdir, cfg: DictConfig):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = SummaryWriter(os.path.join(outdir, f'tb-{str(seed)}'))

    make_meta_task = hydra.utils.instantiate(cfg.meta.task)
    env = make_meta_task(cfg)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = Monitor(
        env,
        filename=f'{outdir}/{seed}',
        info_keywords=('agent_stats', 'cf_agent_stats')
    )
    env.seed(seed)

    if cfg.meta.alg.variant == "exp3s":
        mab = Exp3S(
            nbArms=env.action_space.n,
            alpha=cfg.meta.alg.alpha,  # hparams Graves et al
            gamma=cfg.meta.alg.gamma,
        )
        mab.startGame()
    else:
        raise ValueError('cfg.alg.variant')

    _ = env.reset()
    for i in range(cfg.meta.alg.total_timesteps):
        action = mab.choice()
        _, reward, done, info = env.step(action)
        mab.getReward(action, np.clip(reward, -1.0, 1.0))

        global_step = i * info['play_steps']
        logger.add_scalar(
            "eval/returns", reward, global_step=global_step)

        if done:
            env.reset()
            for arm in range(mab.nbArms):
                logger.add_scalar(
                    f"arm-{arm + 1}/reward", mab.rewards[arm],
                    global_step=global_step)
                logger.add_scalar(
                    f"arm-{arm + 1}/pulls", mab.pulls[arm],
                    global_step=global_step)

    env.close()
    logger.close()


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:
    torch.backends.cudnn.deterministic = cfg.torch.deterministic

    log = logging.getLogger(__name__)

    opponent = cfg.meta.task.evaluator_action

    outdir = os.path.join(
        cfg.run.outdir,
        # cfg.meta.task.id,
        cfg.main.task.id,
        cfg.main.alg.id,
        "all" if opponent == -1 else f"player_{opponent}"
    )
    os.makedirs(outdir, exist_ok=True)

    seeds = range(cfg.run.seed, cfg.run.seed + cfg.run.num_seeds)
    games = {}

    try:
        tmp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass

    with tmp.Pool(
        processes=cfg.thread_pool.size,
        maxtasksperchild=cfg.thread_pool.maxtasks
    ) as ppe:
        for seed in seeds:
            log.info(f"<submit> tscl game with seed: {seed}")
            games[ppe.apply_async(
                play_bandit, (seed, outdir, cfg))] = seed

        for game in games:
            try:
                code = game.get()
                seed = games[game]
                log.info(
                    f'<finished> game with {seed} finished with exit code {code}')
            except Exception as ex:
                log.error(
                    f'<FAIL> game with {seed} with the following exception: \n',
                    traceback.format_exc(),
                    exc_info=ex
                )


if __name__ == '__main__':
    main()
