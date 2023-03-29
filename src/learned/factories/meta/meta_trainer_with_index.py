
from gym.wrappers import TimeLimit

from omegaconf import DictConfig
from ccgm.common.envs.sgt.wrappers import OneHotObservationWrapper
from ccgm.common.envs.utils import AutoResetWrapper

from learned.utils import make_meta_env
from learned.wrappers import (
    FixMetaEvaluatorAction, JointActionObservationWrapper
)


def make_task(
    meta_player_id: int,
    evaluator_action: int,
    episode_time_limit: int,
    num_envs: int
):
    def make_meta_task(
        task_config: DictConfig
    ):
        env = make_meta_env(task_config)
        env = JointActionObservationWrapper(
            env=env,
        )
        # env = FixMetaEvaluatorAction(
        #     env=env,
        #     evaluator_action=evaluator_action
        # )
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
