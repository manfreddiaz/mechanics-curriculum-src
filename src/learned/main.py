import functools
import gym

import numpy as np
import hydra
import torch
from omegaconf import DictConfig
from ccgm.common.algs.core import Agent

from learned.core import MetaTrainingEnvironment
from learned.utils import hydra_custom_resolvers


hydra_custom_resolvers()


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

    return np.mean(episodes_rewards), np.std(episodes_rewards)


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:

    make_agent = hydra.utils.instantiate(cfg.agent)
    _, make_env = hydra.utils.instantiate(cfg.task)
    make_alg = hydra.utils.instantiate(cfg.alg)

    env = MetaTrainingEnvironment(
        agent_fn=functools.partial(
            make_agent,
            hparams=cfg.alg.hparams,
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
    env.seed(0)

    obs = env.reset()
    for i in range(100):
        trainer_action = env.action_space.sample()
        eval_action = env.action_space.sample()

        obs, reward, done, info = env.step([trainer_action, eval_action])
        print(reward)


if __name__ == '__main__':
    main()
