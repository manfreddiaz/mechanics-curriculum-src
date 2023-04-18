
from gym.wrappers import TimeLimit

from omegaconf import DictConfig
from ccgm.common.envs.utils import AutoResetWrapper

from learned.utils import make_meta_env
from learned.wrappers import (
    FixedEvaluatorWrapper, RewardLearningProgressionWrapper,
    TimeFeatureWrapper
)


def make_task(
    id: str,
    evaluator_action: int,
    episode_time_limit: int,
    num_envs: int
):
    def make_meta_task(
        task_config: DictConfig
    ):
        env = make_meta_env(task_config)
        env = FixedEvaluatorWrapper(env, eval_action=evaluator_action)
        env = RewardLearningProgressionWrapper(
            env=env
        )

        if episode_time_limit > 0:
            env = TimeLimit(
                env,
                episode_time_limit
            )
            env = AutoResetWrapper(
                env=env
            )
        return env

    return make_meta_task
