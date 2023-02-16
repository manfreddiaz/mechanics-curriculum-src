import numpy as np
from gym import envs
from gym.envs.registration import register


from ..strategies import (
    NatureMemoryOneStrategy, PrincipalMarkovStrategy
)


PD_NATURE_AVAILABLE_STRATEGIES = dict({
    'tit_for_tat': [
        np.array([1.0, 0.0]),
        np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0]
        ])
    ],
    'always_cooperate': [
        np.array([1.0, 0.0]),
        np.array([
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
    ],
    'always_defect': [
        np.array([0.0, 1.0]),
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ])
    ],
    'win_stay_lose_switch': [
        np.array([1.0, 0.0]),
        np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0]
        ])
    ],
    'extortionate-zd-default': [
        np.array([1.0, 0.0]),
        np.array([
            [7.0 / 9.0, 0.0, 6.0 / 9.0, 0.0],
            [2.0 / 9.0, 1.0, 3.0 / 9.0, 1.0]
        ])
    ]
})


def pd_nature_strategy_factory(strategy_name: str) -> NatureMemoryOneStrategy:

    assert strategy_name in PD_NATURE_AVAILABLE_STRATEGIES
    initial, memory_one = PD_NATURE_AVAILABLE_STRATEGIES[strategy_name]
    return NatureMemoryOneStrategy(
        name=strategy_name,
        initial_action_dist=initial,
        memory_one_action_dist=memory_one
    )


def pd_principal_strategy_factory(
    strategy_name: str
) -> PrincipalMarkovStrategy:

    if "default" == strategy_name:
        return PrincipalMarkovStrategy(
            name=strategy_name,
            payoff_matrix=np.array([
                [3.0, 0.0, 5.0, 1.0],
                [3.0, 5.0, 0.0, 1.0]
            ])
        )
    else:
        raise NotImplementedError()

register(
    id="SIPD/AlwaysCooperate-Default-v0",
    entry_point='ccgm.common.envs.sgt:SequentialMatrixGameEnvironment',
    kwargs={
        'nature_strategy': 'always_cooperate',
        'principal_strategy': 'default',
        'nature_strategy_factory': pd_nature_strategy_factory,
        'principal_strategy_factory': pd_principal_strategy_factory
    }
)
register(
    id="SIPD/AlwaysDefect-Default-v0",
    entry_point='ccgm.common.envs.sgt:SequentialMatrixGameEnvironment',
    kwargs={
        'nature_strategy': 'always_defect',
        'principal_strategy': 'default',
        'nature_strategy_factory': pd_nature_strategy_factory,
        'principal_strategy_factory': pd_principal_strategy_factory
    }
)
register(
    id="SIPD/TitForTat-Default-v0",
    entry_point='ccgm.common.envs.sgt:SequentialMatrixGameEnvironment',
    kwargs={
        'nature_strategy': 'tit_for_tat',
        'principal_strategy': 'default',
        'nature_strategy_factory': pd_nature_strategy_factory,
        'principal_strategy_factory': pd_principal_strategy_factory
    }
)
register(
    id="SIPD/WinStayLoseSwitch-Default-v0",
    entry_point='ccgm.common.envs.sgt:SequentialMatrixGameEnvironment',
    kwargs={
        'nature_strategy': 'win_stay_lose_switch',
        'principal_strategy': 'default',
        'nature_strategy_factory': pd_nature_strategy_factory,
        'principal_strategy_factory': pd_principal_strategy_factory
    }
)
register(
    id="SIPD/ExtortionateZD-Default-v0",
    entry_point='ccgm.common.envs.sgt:SequentialMatrixGameEnvironment',
    kwargs={
        'nature_strategy': 'extortionate-zd-default',
        'principal_strategy': 'default',
        'nature_strategy_factory': pd_nature_strategy_factory,
        'principal_strategy_factory': pd_principal_strategy_factory
    }
)

SPID_STRATEGIES = {
    env_id: env_spec 
        for env_id, env_spec in envs.registry.env_specs.items() if "SIPD" in env_spec.id
}