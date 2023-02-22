from omegaconf import DictConfig

from ccgm.common.algs.ppo import PPO


def make_alg(
    id: str,
    hparams: DictConfig,
    rparams: DictConfig
):
    def make_ppo(
        envs
    ):
        agent = PPO(
            envs=envs,
            rparams=rparams,
            hparams=hparams
        )
        return agent

    return make_ppo