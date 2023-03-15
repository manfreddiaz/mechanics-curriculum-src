from multiprocessing import Event
from typing import Callable, List, Union
import gym
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from .coalitions import Coalition  # , DQN, A2C
# from .envs.sgt.env import SequentialMatrixGameEnvironment
from .strategies import MetaStrategy


class CooperativeMetaGame(gym.Wrapper):

    def __init__(
        self,
        meta_strategy: Coalition,
    ) -> None:
        super().__init__(
            env=meta_strategy._strategy_space[0]
        )
        self._meta_strategy = meta_strategy

    def step(self, action):
        next_state, reward, done, info =  super().step(action)

        info['meta-strategy'] = self.env.spec.id
        if done:
            self.env = self._meta_strategy()

        return next_state, reward, done, info

    def seed(self, seed=None):
        self._meta_strategy.seed(seed)
        return super().seed(seed)
        

class CooperativeMetaGameEnvironment(gym.Env):

    def __init__(
        self,
        strategies: List[str],
        strategies_factory: Callable[[str], Callable]
    ) -> None:
        super().__init__()
        self.strategies = strategies
        self.strategies_factory = strategies_factory

        self.action_space = spaces.Discrete(len(strategies))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=[len(strategies)])

    def step(self, action):
        return super().step(action)


class MetaGameSyncWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        episodes_per_step: int = 1
    ) -> None:
        super().__init__(env)
        self.nature_set = Event()
        self.on_step_end = Event()
        self.episodes_per_steps = episodes_per_step
        self.n_steps = episodes_per_step

    def reset(self):
        self.nature_set.wait(30)
        obs = super().reset()
        self.nature_set.clear()
        return obs

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        if done:
            self.n_steps -= 1
            if self.n_steps == 0:
                self.on_step_end.set()
                self.n_steps = self.episodes_per_steps
            else:
                self.nature_set.set()
        return next_state, reward, done, info


class MetaGameBanditEnvironment(gym.Env):

    def __init__(
        self,
        env: MetaGameSyncWrapper,
        train_strategies: List[str],
        eval_env: gym.Env,
        strategies_factory: Callable[[str], Callable],
        learner: PPO
    ) -> None:
        super().__init__()
        self.strategies = train_strategies
        self.strategies_factory = strategies_factory

        self.env = env
        self.eval_env = eval_env
        self.learner = learner

        self.action_space = spaces.Discrete(len(train_strategies))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=[len(train_strategies)])
        self.state = np.zeros(len(train_strategies))
        self.last_reward = self._reward()
        self.zero_state = np.zeros_like(self.state)

    def reset(self):
        self.state.fill(0)
        return self.state

    def _act(self, action):
        assert self.env.unwrapped.nature_strategy is not None
        self.env.unwrapped.nature_strategy = self.strategies_factory(
            self.strategies[action])

    def _reward(self):
        mean, std = evaluate_policy(
            self.learner, self.eval_env, n_eval_episodes=1,
            deterministic=False
        )
        return mean

    def _obs(self, action):
        self.state[action] += 1
        # obs = (self.state + 1e-5)
        # obs /= sum(obs)
        # return obs
        return self.zero_state

    def step(self, action):
        self._act(action)
        self.env.nature_set.set()
        # self.state[action] += 1
        # print('waiting for ep')
        self.env.on_step_end.wait(30)
        # self.env.nature_set.clear()
        # wait for episode to be produced
        _reward = self._reward()
        reward = _reward - self.last_reward
        self.last_reward = _reward
        # print('ep produced')
        obs = self._obs(action)
        self.env.on_step_end.clear()
        return obs, reward, False, {'freqs': self.state}


if __name__ == '__main__':
    env = SequentialMatrixGameEnvironment(
        nature_strategy='always_cooperate',
        principal_strategy='default',
        nature_strategy_factory=prisioner_dilemma.pd_nature_strategy_factory,
        principal_strategy_factory=prisioner_dilemma.pd_principal_strategy_factory
    )
    env = CooperativeMetaGame(
        env=env,
        nature_strategy=Coalition(
            players=['always_defect', 'tit_for_tat'],
            strategy_factory=prisioner_dilemma.pd_nature_strategy_factory
        ),
        principal_strategy=Coalition(
            players=['default'],
            strategy_factory=prisioner_dilemma.pd_principal_strategy_factory
        )
    )

    for i in range(200):
        obs = env.reset()
        print("nature:", env.unwrapped.nature_strategy)
        print("principal: ", env.unwrapped.principal_strategy)
