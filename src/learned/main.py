import functools
import random
import numpy as np

import gym
import hydra
from omegaconf import DictConfig
import torch

from learned.utils import hydra_custom_resolvers

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


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:
    random.seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    torch.backends.cudnn.deterministic = cfg.torch.deterministic

    make_meta_task = hydra.utils.instantiate(cfg.meta.task)
    make_env = functools.partial(
        make_meta_task,
        cfg
    )
    envs = gym.vector.SyncVectorEnv([
        make_env for i in range(cfg.meta.task.num_envs)
    ])

    trainer, trainer_play_fn, trainer_optim_fn = make_meta_agent(cfg, envs)
    # evaluer, evaluer_play_fn, evaluer_optim_fn = make_meta_agent(cfg, envs)

    obs = envs.reset()
    for i in range(39000):
        trainer_action, t_memory_fn = trainer_play_fn(trainer, obs, i)
        # eval_action, e_memory_fn = evaluer_play_fn(evaluer, obs, i)

        joint_action = np.hstack([[trainer_action], [np.array([1])]])
        next_obs, reward, done, info = envs.step(joint_action)
        t_memory_fn(obs, reward, next_obs, done, info)
        # e_memory_fn(obs, -reward, next_obs, done, info)

        trainer_optim_fn(trainer, envs, i)
        # evaluer_optim_fn(evaluer, envs, i)

        print(
            f"it:{i}, ta: {trainer_action}, ea: {1}, rt: {reward[0]:.1f}")
        if done:
            print('one done')

        obs = next_obs


if __name__ == '__main__':
    main()
