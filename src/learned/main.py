import functools
import gym
import hydra
from omegaconf import DictConfig
import torch

from learned.utils import hydra_custom_resolvers

hydra_custom_resolvers()


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:

    make_meta_task = hydra.utils.instantiate(cfg.meta.task)
    make_env = functools.partial(
        make_meta_task,
        cfg
    )
    envs = gym.vector.SyncVectorEnv([
        make_env for i in range(cfg.meta.task.num_envs)
    ])

    make_agent = hydra.utils.instantiate(cfg.meta.agent)
    agent = make_agent(
        envs=envs,
        hparams=cfg.meta.alg.hparams,
        device=torch.device(cfg.torch.device)
    )

    make_alg = hydra.utils.instantiate(cfg.meta.alg)
    _, _, learn_fn = make_alg(
        logger=None,
        device=torch.device(cfg.torch.device),
        log_every=cfg.run.log_every,
        log_file_format=None
    )

    learn_fn(
        agent=agent,
        envs=envs
    )

    # obs = env.reset()
    # for i in range(100):
    #     trainer_action = env.action_space.sample()
    #     eval_action = env.action_space.sample()

    #     obs, reward, done, info = env.step([trainer_action, eval_action])
    #     print(reward)
    #     if done:
    #         print('one done')


if __name__ == '__main__':
    main()
