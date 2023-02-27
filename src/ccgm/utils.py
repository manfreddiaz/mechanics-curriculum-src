from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable, List, NamedTuple


def form_coalitions(
    players: List, 
    ordered: bool = False, 
    min_order: int = 1,
    max_order: int = None
):
    num_players = len(players)
    max_order = max_order if max_order is not None else num_players
    
    coalesce = permutations if ordered else combinations
    # teams = []
    idx = 0
    for i in range(min_order, max_order + 1):
        for team in coalesce(players, i):
            # teams.append(team)
            coalition = Coalition(
                players=team,
                idx=idx,
                ordered=ordered
            )
            idx += 1
            yield coalition


# def coalition_to_id(team: List):
#     return '+'.join(list(team))


# def id_to_coalition(id: str):
#     return id.split('+')

@dataclass
class Coalition:
    players: List[str]
    idx: int
    ordered: bool
    id: str = None

    def __post_init__(self):
        self.id = Coalition.to_id(self.players)
    
    @classmethod
    def from_id(cls, id: str):
        return Coalition(
            players=id.split('+'),
            idx=None,
            ordered=None,
            id=id
        )

    @classmethod
    def to_id(cls, players) -> 'str':
        return '+'.join(list(players))


@dataclass
class CoalitionalGame:
    players: Iterable
    coalitions: Iterable[Coalition]

    @classmethod
    def make(cls, players: List, ordered: bool)-> 'CoalitionalGame':
        coalitions = form_coalitions(
            players=players, 
            ordered=ordered 
        )

        return CoalitionalGame(players, coalitions)