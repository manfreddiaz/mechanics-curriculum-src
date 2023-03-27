
from gym.wrappers import TimeLimit

from omegaconf import DictConfig
from ccgm.common.envs.sgt.wrappers import OneHotObservationWrapper
from ccgm.common.envs.utils import AutoResetWrapper

from learned.utils import make_meta_env
from learned.wrappers import MetaActionIdentificationWrapper


def make_task(
    meta_player_id: int,
    episode_time_limit: int,
    num_envs: int
):
    def make_meta_task(
        task_config: DictConfig
    ):
        env = make_meta_env(task_config)
        env = MetaActionIdentificationWrapper(
            env=env,
            meta_player_id=meta_player_id
        )
        env = OneHotObservationWrapper(env)
        env = TimeLimit(
            env,
            episode_time_limit
        )
        env = AutoResetWrapper(
            env=env
        )
        return env

    return make_meta_task
