from typing import Callable, List, Union
import numpy as np

from .strategies import MetaStrategy


class Coalition(MetaStrategy):
    def __init__(
        self,
        players: Union[List[Callable], List[str]],
        strategy_factory: Callable[[str], Callable] = None,
        probs: np.array = None,
    ) -> None:
        super().__init__(players, strategy_factory)
        self.rng = np.random.RandomState(0)
        if probs is not None:
            self.probs = probs
        else:
            self.probs = np.ones(len(players)) / len(players)

    def seed(self, seed: int):
        self.rng.seed(seed)

    def __call__(self, **kwargs) -> Callable:
        return self.rng.choice(self._strategy_space, p=self.probs)


class OrderedCoalition(Coalition):

    def __init__(
        self,
        players: Union[List[Callable], List[str]],
        time_limit: int,
        strategy_factory: Callable[[str], Callable] = None,
        probs: np.array = None,
    ) -> None:
        super().__init__(players, strategy_factory)
        self.rng = np.random.RandomState(0)
        self.time_limit = time_limit
        self.time = 0
        if probs is not None:
            self.probs = np.array(probs)
        else:
            self.probs = np.ones(len(players)) / len(players)
        self.segments = np.floor(np.cumsum(self.probs * self.time_limit))

    def seed(self, seed: int):
        self.rng.seed(seed)

    def __call__(self, **kwargs) -> Callable:
        self.time += kwargs.get('num_updates')
        action = np.searchsorted(self.segments, self.time)
        action = min(action, len(self._strategy_space) - 1)
        return self._strategy_space[action]
