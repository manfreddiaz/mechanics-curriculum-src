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

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        info['meta-strategy'] = self.env.spec.id
        if done:
            self.env = self._meta_strategy()

        return next_state, reward, done, info

    def seed(self, seed=None):
        self._meta_strategy.seed(seed)
        return super().seed(seed)
