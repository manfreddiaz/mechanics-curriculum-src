
from gym.wrappers import TimeLimit

from omegaconf import DictConfig
from ccgm.common.envs.utils import AutoResetWrapper
from learned.core import (
    CounterfactualRewardWrapper, CounterfactualSelfPlayWrapper
)
from learned.utils import make_meta_env


def make_task(
    id: str,
    learning_progression: bool,
    meta_player_id: int,
    evaluator_action: int,
    episode_time_limit: int,
    num_envs: int
):
    def make_meta_task(
        task_config: DictConfig
    ):
        env = make_meta_env(task_config, counter_factual=True)
        if episode_time_limit > 0:
            env = TimeLimit(
                env,
                episode_time_limit
            )
            env = CounterfactualSelfPlayWrapper(
                env,
                learning_progression=learning_progression
            )
            env = CounterfactualRewardWrapper(env)
            env = AutoResetWrapper(
                env=env
            )
        return env

    return make_meta_task
