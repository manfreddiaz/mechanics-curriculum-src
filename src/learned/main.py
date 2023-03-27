import hydra
from omegaconf import DictConfig

from learned.utils import hydra_custom_resolvers

hydra_custom_resolvers()


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(
    cfg: DictConfig
) -> None:

    make_meta_task = hydra.utils.instantiate(cfg.meta.task)
    env = make_meta_task(cfg)
    env.seed(cfg.run.seed)

    obs = env.reset()
    for i in range(100):
        trainer_action = env.action_space.sample()
        eval_action = env.action_space.sample()

        obs, reward, done, info = env.step([trainer_action, eval_action])
        print(reward)
        if done:
            print('one done')


if __name__ == '__main__':
    main()
