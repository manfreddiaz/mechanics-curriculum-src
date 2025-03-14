
from typing import Callable, Union

import gym
import numpy as np
from gym.wrappers import TimeLimit

from ccgm.common.envs.utils import AutoResetWrapper

from .strategies import NatureMemoryOneStrategy, PrincipalMarkovStrategy
from .wrappers import OneHotObservationWrapper, SparseRewardWrapper


class SequentialMatrixGameEnvironment(gym.Env):
    def __init__(
        self,
        nature_strategy: Union[NatureMemoryOneStrategy, str],
        principal_strategy: Union[PrincipalMarkovStrategy, str],
        nature_strategy_factory: Callable[
            [str], NatureMemoryOneStrategy] = None,
        principal_strategy_factory: Callable[
            [str], PrincipalMarkovStrategy] = None,
        with_memory: bool = False
    ) -> None:

        super().__init__()

        self.np_random = np.random.RandomState(0)
        self.acc_rewards = 0.0

        self._last_obs = None

        if type(nature_strategy) == str:
            assert nature_strategy_factory is not None
            self.nature_strategy = nature_strategy_factory(nature_strategy)
        elif type(nature_strategy) == NatureMemoryOneStrategy:
            self.nature_strategy = nature_strategy
        else:
            raise NotImplementedError()

        if type(principal_strategy) == str:
            assert principal_strategy_factory is not None
            self.principal_strategy = principal_strategy_factory(
                principal_strategy)
        elif type(principal_strategy) == PrincipalMarkovStrategy:
            self.principal_strategy = principal_strategy
        else:
            raise NotImplementedError()

        self.action_space = gym.spaces.Discrete(
            self.nature_strategy.action_dim)
        self.observation_space = gym.spaces.Discrete(
            self.nature_strategy.action_dim
        )

    def seed(self, seed: int):
        self.np_random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.nature_strategy.seed(seed)
        return

    def reset(self, seed=None, options=None):
        self._last_obs = self.nature_strategy.first()
        self.acc_rewards = 0.0
        return self._last_obs

    def step(self, action):
        next_obs = self.nature_strategy(self._last_obs, action)
        nature_reward, agent_reward = self.principal_strategy(
            self._last_obs, action)
        self.acc_rewards += nature_reward

        self._last_obs = next_obs
        info = dict({
            'nature_reward': nature_reward,
            'nature_acc_reward': self.acc_rewards
        })
        return next_obs, agent_reward, False, info


def make_env(
    env_id: str,
    episode_time_limit: int,
    sparse: bool,
    one_hot: bool
):
    env = gym.make(env_id)
    env = TimeLimit(
        env,
        episode_time_limit
    )
    env = AutoResetWrapper(
        env=env
    )
    if sparse:
        env = SparseRewardWrapper(env)

    if one_hot:
        env = OneHotObservationWrapper(env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
