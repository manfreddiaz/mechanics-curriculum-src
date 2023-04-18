from distutils.util import strtobool
import random
import numpy as np

import gym
import hydra
from omegaconf import DictConfig
import torch

from stable_baselines3.common.monitor import Monitor
from SMPyBandits.Policies import Exp3, Exp3S, UCB, Hedge

from learned.utils import eval_agent, hydra_custom_resolvers


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
    
    epochs = 50000
    make_meta_task = hydra.utils.instantiate(cfg.meta.task)
    env = make_meta_task(cfg)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = Monitor(
        env,
        filename=f'{cfg.meta.task.id}/{cfg.main.task.id}/{cfg.main.alg.id}/{cfg.run.seed}',
        info_keywords=('agent_stats' , 'cf_agent_stats')
    )
    env.seed(cfg.run.seed)

    mab = UCB(
        nbArms=env.action_space.n,
        # alpha=1e-5,
        # gamma=0.05,
    )
    mab.startGame()

    obs = env.reset()
    for i in range(epochs):
        action = mab.choice()
        obs, reward, done, info = env.step(action)
        mab.getReward(action, reward)       
        if done:
            # print("t: ", mab.trusts)
            print("r: ", mab.rewards)
            print("p: ", mab.pulls)
            env.reset()
    

if __name__ == '__main__':
    main()
