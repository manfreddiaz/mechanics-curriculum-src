from typing import Union
import gym

from .coalitions import Coalition


class CooperativeMetaGame(gym.Wrapper):

    def __init__(
        self,
        meta_strategy: Coalition,
    ) -> None:
        super().__init__(
            env=meta_strategy._strategy_space[0]
        )
        self._meta_strategy = meta_strategy
        self.steps = 0

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        info['meta-strategy'] = self.env.spec.id
        self.steps += 1
        if done:
            self.env = self._meta_strategy(num_updates=self.steps)
            self.steps = 0
            self.env.reset()

        return next_state, reward, done, info

    def seed(self, seed=None):
        self._meta_strategy.seed(seed)
        return super().seed(seed)
