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
            # else:
            #     assert isinstance(strategy, Callable)
            self._strategy_space.append(
                strategy
            )

    def __call__(self, **kwargs) -> Callable:
        raise NotImplementedError()




