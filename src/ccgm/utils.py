from itertools import combinations, permutations
from typing import List


def form_coalition(players, order):
    return combinations(players, order)


def form_ordered_coalition(players, order):
    return permutations(players, order)


def form_teams(players: List, ordered: bool = False, min_order: int = 1, max_order: int = None):
    num_players = len(players)
    max_order = max_order if max_order is not None else num_players
    
    coalesce = form_ordered_coalition if ordered else form_coalition
    teams = []
    for i in range(min_order, max_order + 1):
        for team in coalesce(players, i):
            teams.append(team)
    return teams


def team_to_id(team: List):
    return '+'.join(list(team))


def id_to_team(id: str):
    return id.split('+')
