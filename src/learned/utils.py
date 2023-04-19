
import functools
import numpy as np
import gym
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from ccgm.common.algs.core import Agent

from learned.core import (
    MetaTrainingEnvironment, CounterfactualMetaTrainingEnvironment
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


def eval_agent(
    agent: Agent,
    envs: gym.vector.VectorEnv,
    episodes: int,
    device: torch.device
):
    # Adapted from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/evaluation.html # noqa
    policy = agent.policy

    # accumulators
    rewards = np.zeros(envs.num_envs, dtype=np.float32)
    steps = np.zeros(envs.num_envs)
    run_episodes = np.zeros(envs.num_envs)
    episodes_rewards, episodes_lengths = [], []

    next_obs = envs.reset()
    while (run_episodes < episodes).any():
        with torch.no_grad():
            action = policy.predict(torch.tensor(next_obs).to(device))
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards += reward
        steps += 1
        for i in range(envs.num_envs):
            if done[i]:
                run_episodes[i] += 1
                episodes_rewards.append(rewards[i])
                episodes_lengths.append(steps[i])
                rewards[i] = 0.0
                steps[i] = 0

    episodes_rewards = np.array(episodes_rewards)

    return np.mean(episodes_rewards), np.std(episodes_rewards)


def make_meta_env(cfg: DictConfig, counter_factual: bool = False) -> gym.Env:
    make_agent = hydra.utils.instantiate(cfg.main.agent)
    _, make_env = hydra.utils.instantiate(cfg.main.task)
    make_alg = hydra.utils.instantiate(cfg.main.alg)
    if counter_factual:
        env = CounterfactualMetaTrainingEnvironment(
            agent_fn=functools.partial(
                make_agent,
                hparams=cfg.main.alg.hparams,
                device=torch.device(cfg.torch.device)
            ),
            env_fn=make_env,
            alg_fn=functools.partial(
                make_alg,
                logger=None,  # TODO: at some point logging logic has to go
                device=torch.device(cfg.torch.device),
                log_every=cfg.run.log_every,
                log_file_format=None
            ),
            eval_fn=functools.partial(
                eval_agent,
                episodes=5,  # 5 episodes on each vectorized env
                device=torch.device(cfg.torch.device)
            )
        )
    else:
        env = MetaTrainingEnvironment(
            agent_fn=functools.partial(
                make_agent,
                hparams=cfg.main.alg.hparams,
                device=torch.device(cfg.torch.device)
            ),
            env_fn=make_env,
            alg_fn=functools.partial(
                make_alg,
                logger=None,  # TODO: at some point logging logic has to go
                device=torch.device(cfg.torch.device),
                log_every=cfg.run.log_every,
                log_file_format=None
            ),
            eval_fn=functools.partial(
                eval_agent,
                episodes=5,  # 5 episodes on each vectorized env
                device=torch.device(cfg.torch.device)
            )
        )

    return env

