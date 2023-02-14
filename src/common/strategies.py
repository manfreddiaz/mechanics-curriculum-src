from typing import Callable, List, Union
import numpy as np


class MetaStrategy:

    def __init__(
        self,
        strategies: Union[List[Callable], List[str]],
        strategy_factory: Callable[[str], Callable] = None,
    ) -> None:
        assert len(strategies) > 0, "should specify at least one strategy"

        self._strategy_space = []
        for strategy in strategies:
            if isinstance(strategy, str):
                assert strategy_factory is not None
                strategy = strategy_factory(strategy)
            else:
                assert isinstance(strategy, Callable)
            self._strategy_space.append(
                strategy
            )

    def __call__(self, **kwargs) -> Callable:
        raise NotImplementedError()


class ProbabilisticMetaStrategy(MetaStrategy):

    def __init__(
        self,
        strategies: Union[List[Callable], List[str]],
        strategy_factory: Callable[[str], Callable] = None,
        probs: np.array = None,
    ) -> None:
        super().__init__(strategies, strategy_factory)
        self.rng = np.random.RandomState(0)
        if probs is not None:
            self.probs = probs
        else:
            self.probs = np.ones(len(strategies)) / len(strategies)

    def seed(self, seed: int):
        self.rng.seed(seed)

    def __call__(self, **kwargs) -> Callable:
        return self.rng.choice(self._strategy_space, p=self.probs)


class TimeDependentMetaStrategy(MetaStrategy):

    def __init__(
        self,
        strategies: Union[List[Callable], List[str]],
        time_limit: int,
        strategy_factory: Callable[[str], Callable] = None,
        probs: np.array = None,
    ) -> None:
        super().__init__(strategies, strategy_factory)
        self.rng = np.random.RandomState(0)
        if probs is not None:
            self.probs = probs
        else:
            self.probs = np.ones(len(strategies)) / len(strategies)

    def seed(self, seed: int):
        self.rng.seed(seed)

    def __call__(self, **kwargs) -> Callable:
        # TODO
        return self.rng.choice(self._strategy_space, p=self.probs)

