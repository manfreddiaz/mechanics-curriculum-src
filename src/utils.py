from itertools import combinations
from typing import List

import gym

from m3g.examples.games.env import SequentialMatrixGameEnvironment
from common.games import (
    CooperativeMetaGame, ProbabilisticMetaStrategy
)
from m3g.examples.games.impl.prisioner_dilemma import (
    pd_nature_strategy_factory,
    pd_principal_strategy_factory
)
from m3g.examples.games.wrappers import (
    OneHotObservationWrapper, SparseRewardWrapper
)


def form_team(players, order):
    return combinations(players, order)


def form_teams(players: List, min_order: int = 1, max_order: int = None):
    num_players = len(players)
    max_order = max_order if max_order is not None else num_players
    teams = []
    for i in range(min_order, max_order + 1):
        for team in form_team(players, i):
            teams.append(team)
    return teams


def make_cooperative_env(
    team: List,
    sparse: bool = True,
    one_hot: bool = True
) -> 'CooperativeMetaGame':
    env = SequentialMatrixGameEnvironment(
        nature_strategy='always_cooperate',
        principal_strategy='default',
        nature_strategy_factory=pd_nature_strategy_factory,
        principal_strategy_factory=pd_principal_strategy_factory
    )

    env = CooperativeMetaGame(
        env=env,
        nature_strategy=ProbabilisticMetaStrategy(
            strategies=team,
            strategy_factory=pd_nature_strategy_factory
        ),
        principal_strategy=ProbabilisticMetaStrategy(
            strategies=['default'],
            strategy_factory=pd_principal_strategy_factory
        )
    )

    env = gym.wrappers.TimeLimit(env, 200)
    if sparse:
        env = SparseRewardWrapper(env)
    if one_hot:
        env = OneHotObservationWrapper(env)

    return env


def team_to_id(team: List):
    return '+'.join(list(team))


def id_to_team(id: str):
    return id.split('+')
