import numpy as np
from gym.envs.registration import register

from ..strategies import (
    NatureMemoryOneStrategy, PrincipalMarkovStrategy
)


RPS_NATURE_AVAILABLE_STRATEGIES = dict({
    "nash": [
        np.ones(3) / 3.0,
        np.ones(shape=(3, 9)) / 3.0
    ]
})


def rps_principal_strategy_factory(strategy_name: str):
    if "default" == strategy_name:
        return PrincipalMarkovStrategy(
            payoff_matrix=np.array([
                [0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0],
                [0.0,  1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0]
            ])
        )
    else:
        raise NotImplementedError()


def rps_nature_strategy_factory(strategy_name: str) -> NatureMemoryOneStrategy:
    assert strategy_name in RPS_NATURE_AVAILABLE_STRATEGIES
    return NatureMemoryOneStrategy(
        *RPS_NATURE_AVAILABLE_STRATEGIES[strategy_name]
    )


register(
    id="RockPaperScissor-Nash-Default-v0",
    entry_point='m3g.examples.games.env:SequentialMatrixGameEnvironment',
    max_episode_steps=200,
    kwargs={
        'nature_strategy': 'nash',
        'principal_strategy': 'default',
        'nature_strategy_factory': rps_nature_strategy_factory,
        'principal_strategy_factory': rps_principal_strategy_factory
    }
)
